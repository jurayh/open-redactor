"""Framework-neutral agent tool adapter.

The definitions use the common function-calling shape accepted by OpenAI,
LangChain-style adapters, and many agent frameworks. MCP is preferred when
the framework supports it; this file is the small fallback for direct calls.
"""

from __future__ import annotations

import json
from typing import Any

from open_redactor.mcp_server import TOOLS, call_tool

FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["inputSchema"],
        },
    }
    for tool in TOOLS
]


def dispatch(name: str, arguments_json: str) -> str:
    """Execute a function call and return the JSON text agents expect."""
    arguments: dict[str, Any] = json.loads(arguments_json or "{}")
    return json.dumps(call_tool(name, arguments), indent=2)
