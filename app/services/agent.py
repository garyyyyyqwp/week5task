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

        # When an image is present, use the multimodal (vision) model for
        # EVERY step and keep the image in context the whole time. The main
        # model (glm-4.6v) is itself multimodal, so there is no reason to
        # strip the image after step 1 — doing so would make the model
        # "forget" the picture as soon as it calls any tool (e.g. calculator),
        # which is fatal for step-by-step problem explanation.
        step_model = VISION_MODEL if image_url else default_model
        step_messages = messages

        try:
            from app.services.llm import _retry_on_rate_limit
            response = await _retry_on_rate_limit(
                client.chat.completions.create(
                    model=step_model,
                    messages=step_messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.3,
                ),
                operation=f"Agent step {step_num}",
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

        choice = response.choices[0]
        assistant_message = choice.message

        # Extract thought (content in the assistant message)
        # Clean up: LLM sometimes leaks tool_calls metadata into the content field
        raw_thought = assistant_message.content or ""
        thought = _clean_thought(raw_thought)
        if thought:
            yield {
                "event": "thought",
                "data": json.dumps(
                    {"step": step_num, "thought": thought},
                    ensure_ascii=False,
                ),
            }

        # Check if LLM wants to call tools
        if assistant_message.tool_calls:
            # Append assistant message with tool calls to conversation history
            # We need to serialize the message properly for the API
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            })

            for tool_call in assistant_message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {"error": "参数解析失败"}

                step = AgentStep(
                    step_number=step_num,
                    thought=thought,
                    action_name=fn_name,
                    action_input=fn_args,
                )

                # Yield action event
                yield {
                    "event": "action",
                    "data": json.dumps(
                        {"step": step_num, "tool": fn_name, "input": fn_args},
                        ensure_ascii=False,
                    ),
                }

                # Execute tool
                observation = await execute_tool(fn_name, fn_args)
                step.observation = observation
                steps.append(step)

                # Yield observation event (truncate for display)
                display_obs = observation[:500] + "..." if len(observation) > 500 else observation
                yield {
                    "event": "observation",
                    "data": json.dumps(
                        {"step": step_num, "tool": fn_name, "result": display_obs},
                        ensure_ascii=False,
                    ),
                }

                # Append tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation,
                })
        else:
            # No tool calls — this is the Final Answer
            final_answer = assistant_message.content or ""
            break
    else:
        # Max steps reached without final answer
        hit_max_steps = True
        if not final_answer:
            final_answer = "抱歉，我无法在限定步骤内完成此问题。请尝试简化问题或增加步骤限制。"

    # Yield final answer
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
