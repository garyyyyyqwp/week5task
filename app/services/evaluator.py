"""LLM-as-Judge Evaluator for Agent Performance.

Evaluates agent runs across 4 dimensions using the same tool_choice="required"
pattern as Week 3's judge.py. Designed for comparing ReAct prompt templates.
"""

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.services.llm import get_client, get_model
from app.utils.config import AGENT_MAX_STEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge Schema
# ---------------------------------------------------------------------------

class AgentJudgeVerdict(BaseModel):
    """Structured evaluation of an agent run."""

    task_completion: int = Field(
        ..., ge=1, le=5,
        description="任务完成度(1-5): Agent是否成功回答了问题",
    )
    tool_selection_accuracy: int = Field(
        ..., ge=1, le=5,
        description="工具选择准确度(1-5): Agent是否选择了正确的工具",
    )
    hallucination_rate: int = Field(
        ..., ge=1, le=5,
        description="信息准确性(1-5): Agent是否基于工具结果回答(5=无幻觉)",
    )
    reasoning_quality: int = Field(
        ..., ge=1, le=5,
        description="推理质量(1-5): Agent的推理链是否逻辑清晰",
    )

    @property
    def weighted_score(self) -> float:
        """Compute weighted average score."""
        return (
            self.task_completion * 0.35
            + self.tool_selection_accuracy * 0.20
            + self.hallucination_rate * 0.25
            + self.reasoning_quality * 0.20
        )


# Tool definition for structured LLM output
_JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_agent",
        "description": "评估Agent的表现，给出各维度的评分",
        "parameters": {
            "type": "object",
            "properties": {
                "task_completion": {
                    "type": "integer",
                    "description": "任务完成度(1-5): 1=完全未回答, 5=完美回答",
                },
                "tool_selection_accuracy": {
                    "type": "integer",
                    "description": "工具选择准确度(1-5): 1=全部选错, 5=全部选对",
                },
                "hallucination_rate": {
                    "type": "integer",
                    "description": "信息准确性(1-5): 1=大量编造, 5=完全基于事实",
                },
                "reasoning_quality": {
                    "type": "integer",
                    "description": "推理质量(1-5): 1=混乱, 5=清晰有逻辑",
                },
            },
            "required": [
                "task_completion",
                "tool_selection_accuracy",
                "hallucination_rate",
                "reasoning_quality",
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Judge Functions
# ---------------------------------------------------------------------------

async def evaluate_agent_run(
    question: str,
    steps: list[dict[str, Any]],
    final_answer: str,
    expected_answer: str | None = None,
) -> AgentJudgeVerdict | None:
    """Evaluate a single agent run using LLM-as-Judge.

    Args:
        question: The original question.
        steps: List of agent step dicts with thought/action/observation.
        final_answer: The agent's final answer.
        expected_answer: Optional ground truth for reference.

    Returns:
        AgentJudgeVerdict if successful, None on failure.
    """
    client = get_client()
    model = get_model()

    # Build evaluation context
    steps_text = ""
    for i, step in enumerate(steps, 1):
        parts = [f"第{i}步:"]
        if step.get("thought"):
            parts.append(f"  思考: {step['thought']}")
        if step.get("action_name"):
            parts.append(f"  行动: {step['action_name']}({json.dumps(step.get('action_input', {}), ensure_ascii=False)})")
        if step.get("observation"):
            obs = step["observation"]
            if len(obs) > 300:
                obs = obs[:300] + "..."
            parts.append(f"  观察: {obs}")
        steps_text += "\n".join(parts) + "\n\n"

    if not steps_text:
        steps_text = "Agent没有调用任何工具，直接给出了回答。"

    expected_section = ""
    if expected_answer:
        expected_section = f"\n【参考答案】\n{expected_answer}\n"

    eval_prompt = (
        f"请评估以下AI Agent的表现。\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【Agent推理过程】\n{steps_text}\n"
        f"【Agent最终回答】\n{final_answer}\n"
        f"{expected_section}\n"
        f"请从以下4个维度评分(1-5分)：\n"
        f"1. 任务完成度: Agent是否成功回答了问题？\n"
        f"2. 工具选择准确度: Agent是否选择了合适的工具？\n"
        f"3. 信息准确性: Agent的回答是否基于工具返回的实际结果（5=无幻觉）？\n"
        f"4. 推理质量: Agent的推理链是否逻辑清晰、步骤合理？"
    )

    try:
        from app.services.llm import _retry_on_rate_limit
        response = await _retry_on_rate_limit(
            client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个AI系统评估专家。请根据Agent的推理过程和最终回答，客观评估其表现。",
                    },
                    {"role": "user", "content": eval_prompt},
                ],
                tools=[_JUDGE_TOOL],
                tool_choice={"type": "function", "function": {"name": "evaluate_agent"}},
                temperature=0.1,
            ),
            operation="Evaluator judge",
        )

        # Parse structured output
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        return AgentJudgeVerdict(**args)

    except Exception as e:
        logger.error("Agent evaluation failed: %s", e)
        return None


async def evaluate_agent_batch(
    runs: list[dict[str, Any]],
    concurrency: int = 3,
) -> list[AgentJudgeVerdict | None]:
    """Evaluate multiple agent runs concurrently.

    Args:
        runs: List of dicts with keys: question, steps, final_answer, expected_answer(optional).
        concurrency: Maximum concurrent evaluations.

    Returns:
        List of AgentJudgeVerdict (or None on failure) for each run.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _eval_with_sem(run: dict[str, Any]) -> AgentJudgeVerdict | None:
        async with sem:
            return await evaluate_agent_run(
                question=run["question"],
                steps=run["steps"],
                final_answer=run["final_answer"],
                expected_answer=run.get("expected_answer"),
            )

    tasks = [_eval_with_sem(run) for run in runs]
    return list(await asyncio.gather(*tasks, return_exceptions=False))
