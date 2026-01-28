# tools/registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: Dict[str, Any]          # 给 LLM 看：参数结构
    func: Callable[..., Any]        # 真正执行的函数


TOOLS: Dict[str, ToolSpec] = {}


def register_tool(name: str, description: str, schema: Dict[str, Any]):
    """
    用装饰器注册工具：
    @register_tool(...)
    def my_tool(...): ...
    """
    def decorator(func: Callable[..., Any]):
        TOOLS[name] = ToolSpec(
            name=name,
            description=description,
            schema=schema,
            func=func,
        )
        return func
    return decorator


def get_tool(name: str) -> Optional[ToolSpec]:
    return TOOLS.get(name)


def list_tools_for_prompt() -> str:
    """把工具列表拼成 prompt 片段"""
    lines = []
    for t in TOOLS.values():
        lines.append(f"- {t.name}: {t.description}\n  args_schema: {t.schema}")
    return "\n".join(lines)
