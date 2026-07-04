"""ReAct Agent Engine — the core reasoning-action loop.

Implements the ReAct (Reasoning + Acting) pattern using OpenAI Function Calling:
  1. LLM receives the conversation with tool definitions
  2. If LLM generates tool_calls → execute tools → append results → continue
  3. If LLM generates content without tool_calls → that's the Final Answer
  4. Safety: break at max_steps to prevent infinite loops

The streaming version yields SSE events for each step, enabling real-time
visualization of the agent's reasoning process.
"""

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from app.services.llm import get_client, get_model
from app.services.tools import TOOL_DEFINITIONS, execute_tool
from app.services.prompts import REACT_PROMPT_TEMPLATES
from app.services.sessions import AgentStep, AgentResult, save_session
from app.utils.config import AGENT_MAX_STEPS, VISION_MODEL

logger = logging.getLogger(__name__)


def _clean_thought(raw: str) -> str:
    """Clean LLM thought text by removing leaked tool_calls metadata.

    Some LLMs (e.g., glm-4-flash) include raw tool_call JSON inside the
    content field, which looks like:
      {"index":0,"finish_reason":"tool_calls","delta":{...},"tool_calls":[...]}
    We strip these artifacts to keep the UI clean.
    """
    import re

    # Remove JSON blocks that look like tool_calls metadata
    cleaned = re.sub(
        r'\{[^{}]*"tool_calls"\s*:\s*\[.*?\][^{}]*\}',
        '',
        raw,
        flags=re.DOTALL,
    )
    # Remove lines that are pure JSON metadata
    cleaned = re.sub(
        r'^.*"finish_reason"\s*:\s*"tool_calls".*$',
        '',
        cleaned,
        flags=re.MULTILINE,
    )
    # Collapse excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


async def run_agent_stream(
    question: str,
    session_id: str,
    template: str = "basic",
    max_steps: int | None = None,
    image_url: str | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[dict]:
    """Run the ReAct agent loop, yielding SSE events for each step.

    SSE event types:
      - thought: Agent's reasoning text for this step
      - action: Tool name and input arguments
      - observation: Tool execution result
      - answer: Final answer text
      - done: Session summary with trace stats

    Args:
        question: User's question.
        session_id: Unique session identifier.
        template: ReAct prompt template name (basic/structured/self_correcting).
        max_steps: Maximum number of reasoning steps (default from config).
        image_url: Optional image URL for image analysis scenarios.
        history: Optional conversation history (user/assistant message pairs)
                 for multi-turn context. Each dict: {"role": "user"|"assistant",
                 "content": str | list}.

    Yields:
        Dict with "event" and "data" keys for SSE formatting.
    """
    max_steps = max_steps or AGENT_MAX_STEPS
    client = get_client()
    default_model = get_model()

    system_prompt = REACT_PROMPT_TEMPLATES.get(template, REACT_PROMPT_TEMPLATES["basic"])

    # Build user message — images go directly into structured content
    # so the multimodal model (glm-4.6v) can SEE the image
    # instead of reading a text description from a separate vision tool.
    user_content: list[dict] = [{"type": "text", "text": question}]
    if image_url:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url},
        })

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    steps: list[AgentStep] = []
    step_num = 0
    final_answer = ""
    hit_max_steps = False

    while step_num < max_steps:
        step_num += 1

        step_model = VISION_MODEL if image_url else default_model
        step_messages = messages

        # Use streaming to enable token-by-token output for the final answer
        try:
            stream = await client.chat.completions.create(
                model=step_model,
                messages=step_messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.3,
                stream=True,
            )
        except Exception as e:
            logger.error("LLM call failed at step %d: %s", step_num, e)
            yield {
                "event": "thought",
                "data": json.dumps(
                    {"step": step_num, "thought": f"⚠️ LLM调用失败: {str(e)}"},
                    ensure_ascii=False,
                ),
            }
            final_answer = "抱歉，AI服务暂时不可用，请稍后重试。"
            break

        # Stream response in real-time.
        # Strategy: yield answer_chunk events for content as it arrives.
        # If tool_calls appear, switch to "tool mode" — treat content as thought.
        collected_content = ""
        collected_tool_calls: dict[int, dict] = {}
        answer_chunks_yielded = False

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Accumulate tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        collected_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            collected_tool_calls[idx]["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            collected_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

            # Accumulate and stream content tokens
            if delta.content:
                collected_content += delta.content
                # Only yield answer_chunk if no tool_calls have been seen
                if not collected_tool_calls:
                    answer_chunks_yielded = True
                    yield {
                        "event": "answer_chunk",
                        "data": json.dumps(
                            {"chunk": delta.content},
                            ensure_ascii=False,
                        ),
                    }

        if collected_tool_calls:
            # Tool-calling step — content is thought, not answer
            raw_thought = _clean_thought(collected_content)

            # If we accidentally streamed answer_chunks, send a reset event
            if answer_chunks_yielded:
                yield {
                    "event": "answer_reset",
                    "data": json.dumps({"reason": "tool_call"}, ensure_ascii=False),
                }

            if raw_thought:
                yield {
                    "event": "thought",
                    "data": json.dumps(
                        {"step": step_num, "thought": raw_thought},
                        ensure_ascii=False,
                    ),
                }

            tool_calls_list = [
                {
                    "id": collected_tool_calls[i]["id"],
                    "type": "function",
                    "function": collected_tool_calls[i]["function"],
                }
                for i in sorted(collected_tool_calls.keys())
            ]

            messages.append({
                "role": "assistant",
                "content": collected_content or None,
                "tool_calls": tool_calls_list,
            })

            for tc in tool_calls_list:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {"error": "参数解析失败"}

                step = AgentStep(
                    step_number=step_num,
                    thought=raw_thought,
                    action_name=fn_name,
                    action_input=fn_args,
                )

                yield {
                    "event": "action",
                    "data": json.dumps(
                        {"step": step_num, "tool": fn_name, "input": fn_args},
                        ensure_ascii=False,
                    ),
                }

                observation = await execute_tool(fn_name, fn_args)
                step.observation = observation
                steps.append(step)

                display_obs = observation[:500] + "..." if len(observation) > 500 else observation
                yield {
                    "event": "observation",
                    "data": json.dumps(
                        {"step": step_num, "tool": fn_name, "result": display_obs},
                        ensure_ascii=False,
                    ),
                }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": observation,
                })
        else:
            # No tool calls — content was streamed as answer_chunk events
            final_answer = collected_content
            break
    else:
        # Max steps reached without final answer
        hit_max_steps = True
        if not final_answer:
            final_answer = "抱歉，我无法在限定步骤内完成此问题。请尝试简化问题或增加步骤限制。"

    # Yield final complete answer (signals end of streaming)
    yield {
        "event": "answer",
        "data": json.dumps(
            {"answer": final_answer, "hit_max_steps": hit_max_steps},
            ensure_ascii=False,
        ),
    }

    # Save session trace
    agent_result = AgentResult(
        session_id=session_id,
        question=question,
        steps=steps,
        final_answer=final_answer,
        total_steps=len(steps),
        template_used=template,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_session(agent_result)

    # Yield done event
    yield {
        "event": "done",
        "data": json.dumps(
            {
                "session_id": session_id,
                "total_steps": len(steps),
                "template": template,
                "hit_max_steps": hit_max_steps,
            },
            ensure_ascii=False,
        ),
    }


async def run_agent_sync(
    question: str,
    session_id: str,
    template: str = "basic",
    max_steps: int | None = None,
    image_url: str | None = None,
    history: list[dict] | None = None,
) -> AgentResult:
    """Run the agent synchronously (collect all events and return final result).

    Useful for testing and non-streaming use cases.

    Args:
        question: User's question.
        session_id: Unique session identifier.
        template: ReAct prompt template name.
        max_steps: Maximum number of reasoning steps.
        image_url: Optional image URL.
        history: Optional conversation history for multi-turn context.

    Returns:
        AgentResult with full reasoning trace.
    """
    async for event in run_agent_stream(
        question=question,
        session_id=session_id,
        template=template,
        max_steps=max_steps,
        image_url=image_url,
        history=history,
    ):
        # Just consume all events — the session is saved in run_agent_stream
        pass

    from app.services.sessions import get_session
    return get_session(session_id)  # type: ignore
