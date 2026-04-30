"""Shared primitives for application-side LLM tools."""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class Tool(ABC):
    """Base interface for tools that enrich an inference request."""

    name: str

    @abstractmethod
    async def close(self) -> None:
        """Release resources held by the tool."""


class ToolRegistry:
    """Small registry so the inference engine can look up enabled tools by name."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    async def close(self) -> None:
        for tool in self._tools.values():
            await tool.close()
