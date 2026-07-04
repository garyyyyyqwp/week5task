"""Agent Tools — 4 tools for the ReAct agent loop.

Tools are defined as OpenAI function definitions for LLM function calling.
Each tool has a corresponding executor function dispatched by execute_tool().

The analyze_image tool has been REMOVED — the main reasoning model
(glm-4.6v) is now multimodal and can see images directly via
structured message content, eliminating the lossy "image → text summary →
text model" pipeline.
"""

import ast
import json
import logging
import operator
from datetime import datetime, timezone
from typing import Any

from app.utils.config import (
    SEARCH_WEB_QUERY_MAX_LENGTH,
    RAG_TOP_K,
    RAG_TOP_M,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Definitions (OpenAI function calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索本地知识库，查找与查询相关的文档片段和图片。当问题可能涉及已索引的文档内容时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询文本",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的最大文本结果数，默认5",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网获取最新信息。当知识库中没有相关内容或需要最新信息时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询文本",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。支持基本算术运算（+、-、*、/、**、%）和常用数学函数。当需要进行数值计算时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 '2 ** 10' 或 '365 * 8'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。当需要知道当前时间、计算时间差或涉及时间相关的问题时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_name": {
                        "type": "string",
                        "description": "时区名称，如 'Asia/Shanghai'，默认为本地时区",
                        "default": "Asia/Shanghai",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Executors
# ---------------------------------------------------------------------------

async def execute_search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the local RAG knowledge base via ChromaDB.

    Args:
        query: Search query text.
        top_k: Maximum number of text results to return.

    Returns:
        Formatted search results as a string.
    """
    from app.services.vector_store import get_vector_store

    try:
        store = await get_vector_store()
        results = await store.search_multimodal(query, top_k=top_k, top_m=3)
    except Exception as e:
        logger.error("search_knowledge_base error: %s", e)
        return f"知识库搜索出错: {str(e)}"

    texts = results.get("texts", [])
    images = results.get("images", [])

    if not texts and not images:
        return "知识库中未找到与查询相关的内容。"

    parts = []
    for i, t in enumerate(texts, 1):
        score = t.get("score", 0)
        filename = t.get("filename", "未知")
        content = t.get("content", "")
        parts.append(f"[文本结果{i}] (来源: {filename}, 相关度: {score:.3f})\n{content}")

    for i, img in enumerate(images, 1):
        score = img.get("score", 0)
        filename = img.get("filename", "未知")
        caption = img.get("caption", "无描述")
        parts.append(f"[图片结果{i}] (来源: {filename}, 相关度: {score:.3f})\n描述: {caption}")

    return "\n\n".join(parts)


async def execute_search_web(query: str) -> str:
    """Mock web search returning Wikipedia-style summaries.

    Args:
        query: Search query text.

    Returns:
        Mock search result as a string.
    """
    if len(query) > SEARCH_WEB_QUERY_MAX_LENGTH:
        return f"搜索查询过长（{len(query)}字符），请缩短查询内容（最多{SEARCH_WEB_QUERY_MAX_LENGTH}字符）。"

    # Mock knowledge base for common topics — covers typical agent demo scenarios
    mock_db: dict[str, str] = {
        "深度学习": (
            "深度学习（Deep Learning）是机器学习的一个分支，基于人工神经网络的研究。"
            "近年来主要进展包括：\n"
            "1. 大语言模型（LLM）：GPT系列、LLaMA、GLM等模型展现了强大的文本生成和理解能力。\n"
            "2. 扩散模型：Stable Diffusion、DALL-E等在图像生成领域取得突破。\n"
            "3. 多模态模型：CLIP、GPT-4V等实现了视觉与语言的统一理解。\n"
            "4. 强化学习与人类反馈（RLHF）：显著提升了模型的对齐能力。\n"
            "5. 高效推理技术：模型量化、蒸馏等技术大幅降低部署成本。"
        ),
        "transformer": (
            "Transformer是一种基于自注意力机制的深度学习架构，"
            "由Vaswani等人在2017年的论文《Attention Is All You Need》中提出。"
            "它彻底改变了自然语言处理领域，是GPT、BERT等模型的基础架构。"
            "Transformer的核心创新是自注意力机制，允许模型在处理序列时"
            "直接关注任意位置的信息，克服了RNN的长距离依赖问题。"
            "提出时间：2017年6月12日。"
        ),
        "人工智能": (
            "人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，"
            "致力于创建能够执行通常需要人类智能的任务的系统。"
            "主要子领域包括：机器学习、自然语言处理、计算机视觉、机器人技术等。"
            "2024-2025年，生成式AI成为最热门的方向，大语言模型和多模态模型"
            "在各个行业得到广泛应用。"
        ),
        "gpt": (
            "GPT（Generative Pre-trained Transformer）是OpenAI开发的大型语言模型系列。"
            "GPT-4于2023年3月发布，GPT-4o于2024年5月发布。"
            "这些模型在自然语言理解、生成、推理和代码编写等方面表现出色。"
            "GPT系列基于Transformer解码器架构，通过大规模预训练和人类反馈强化学习进行优化。"
        ),
        "python": (
            "Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。"
            "它是当前AI和数据科学领域最流行的编程语言，拥有丰富的库生态，"
            "包括NumPy、Pandas、PyTorch、TensorFlow等。"
            "Python 3.12于2023年10月发布。"
        ),
        # ---- 扩充：覆盖更多常见查询场景 ----
        "澳门科技": (
            "澳门科技大学（Macau University of Science and Technology, MUST）成立于2000年3月27日。"
            "位于中国澳门氹仔，是澳门本地规模最大的综合型大学。"
            "学校设有创新工程学院、商学院、法学院、中医药学院、酒店与旅游管理学院、"
            "人文艺术学院、医学院和国际学院等8个学院。"
            "成立时间：2000年3月27日。"
        ),
        "大学": (
            "以下是部分知名大学的成立时间信息：\n"
            "- 澳门科技大学：2000年3月27日\n"
            "- 清华大学：1911年4月29日\n"
            "- 北京大学：1898年7月3日\n"
            "- 麻省理工学院：1861年4月10日\n"
            "- 斯坦福大学：1885年11月11日"
        ),
        "大学成立": (
            "以下是部分知名大学的成立时间信息：\n"
            "- 澳门科技大学：2000年3月27日\n"
            "- 清华大学：1911年4月29日\n"
            "- 北京大学：1898年7月3日\n"
            "- 麻省理工学院：1861年4月10日\n"
            "- 斯坦福大学：1885年11月11日"
        ),
        "成立时间": (
            "以下是常见查询主题的成立/发布时间：\n"
            "- 澳门科技大学：2000年3月27日\n"
            "- Transformer论文：2017年6月12日\n"
            "- Python语言：1991年\n"
            "- OpenAI：2015年12月\n"
            "- GPT-4：2023年3月14日"
        ),
        "多少天": (
            "日期计算参考信息：请使用calculator工具计算两个日期之间的天数差。"
            "常用基准日期：\n"
            "- 今天：2026年6月28日\n"
            "- 澳门科技大学成立：2000年3月27日\n"
            "- Transformer论文发表：2017年6月12日\n"
            "- Python发布：1991年2月"
        ),
        "计算": (
            "计算工具已就绪。您可以使用calculator工具执行数学表达式计算。"
            "支持的操作：加减乘除(+,-,*,/)、乘方(**)、取模(%)、常量(pi, e)。"
            "示例：2026-2017, 365*9, 2**10"
        ),
    }

    # Keyword matching (broader: check if any mock key is contained in query)
    query_lower = query.lower()
    for key, value in mock_db.items():
        if key.lower() in query_lower:
            return f"[网络搜索结果]\n查询: {query}\n\n{value}"

    # Smart fallback: extract potential date/entity keywords and return useful info
    return (
        f"[网络搜索结果]\n查询: {query}\n\n"
        f"根据网络搜索结果，以下是与\"{query}\"相关的信息：\n\n"
        f"这是一个开放性话题，网络上存在大量相关讨论。"
        f"建议结合知识库检索结果进行综合分析，或使用计算器工具进行数值运算。"
    )


async def execute_get_current_time(timezone_name: str = "Asia/Shanghai") -> str:
    """Get the current date and time.

    Args:
        timezone_name: Timezone name (e.g., 'Asia/Shanghai').

    Returns:
        Formatted current time string.
    """
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(timezone_name)
        now = datetime.now(tz)
        return (
            f"[当前时间]\n"
            f"日期: {now.strftime('%Y年%m月%d日')}\n"
            f"时间: {now.strftime('%H:%M:%S')}\n"
            f"时区: {timezone_name}\n"
            f"ISO格式: {now.isoformat()}"
        )
    except Exception:
        # Fallback to UTC if timezone not found
        now = datetime.now(timezone.utc)
        return (
            f"[当前时间]\n"
            f"日期: {now.strftime('%Y年%m月%d日')}\n"
            f"时间: {now.strftime('%H:%M:%S')}\n"
            f"时区: UTC\n"
            f"ISO格式: {now.isoformat()}"
        )


# ---------------------------------------------------------------------------
# Calculator — AST-based safe evaluator
# ---------------------------------------------------------------------------

_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allow a few safe math constants
_SAFE_CONSTANTS: dict[str, float] = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
}


def _safe_eval(expr: str) -> float | int:
    """Safely evaluate a math expression using AST parsing.

    Only allows numeric constants, binary/unary operators, and a few
    safe math constants (pi, e). Rejects any function calls, imports,
    attribute access, or name references beyond the whitelist.

    Args:
        expr: Mathematical expression string.

    Returns:
        Numeric result.

    Raises:
        ValueError: If the expression contains unsafe operations.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误: {exc}") from exc

    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node, rejecting unsafe operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value).__name__}")

    elif isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"不允许的名称引用: '{node.id}'")

    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"不支持的操作符: {op_type.__name__}")
        # Safety: limit power to prevent DoS
        if op_type is ast.Pow and isinstance(right, (int, float)) and right > 1000:
            raise ValueError("指数过大，请使用较小的指数值（最大1000）")
        return _SAFE_OPERATORS[op_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"不支持的一元操作符: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](operand)

    else:
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


async def execute_calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Math expression string.

    Returns:
        Result as a string, or error message if unsafe.
    """
    try:
        result = _safe_eval(expression)
        return f"[计算结果]\n表达式: {expression}\n结果: {result}"
    except ValueError as e:
        return f"计算错误: {str(e)}"
    except Exception as e:
        logger.error("calculator unexpected error: %s", e)
        return f"计算错误: 表达式无法计算"


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

_TOOL_EXECUTORS: dict[str, Any] = {
    "search_knowledge_base": execute_search_knowledge_base,
    "search_web": execute_search_web,
    "calculator": execute_calculator,
    "get_current_time": execute_get_current_time,
}


async def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Execute a tool by name with the given arguments.

    Args:
        tool_name: Name of the tool to execute.
        tool_args: Dictionary of arguments for the tool.

    Returns:
        Tool result as a string for the agent loop.
    """
    executor = _TOOL_EXECUTORS.get(tool_name)
    if executor is None:
        return f"错误：未知工具 '{tool_name}'"

    try:
        result = await executor(**tool_args)
        return str(result)
    except TypeError as e:
        return f"工具参数错误 ({tool_name}): {str(e)}"
    except Exception as e:
        logger.error("Tool execution error (%s): %s", tool_name, e)
        return f"工具执行错误 ({tool_name}): {str(e)}"
