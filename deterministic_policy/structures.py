from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    output: Optional[str] = None
    timestamp: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_dict(self):
        return {
            "name": self.name,
            "arguments": self.arguments,
            "output": self.output,
            "timestamp": self.timestamp,
            "tool_call_id": self.tool_call_id,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            arguments=data["arguments"],
            output=data.get("output"),
            timestamp=data.get("timestamp"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class DFGSource:
    content: str
    source_name: str  # "user_prompt", "system_prompt", or tool_name
    source_index: int  # Index in the sequence (0 for prompt, 1..N for tools)


@dataclass
class DFGNode:
    tool_name: str
    arg_name: str
    arg_value: Any
    sources: List[Dict[str, Any]]  # {content:..., source_name:...}
    tool_call_id: Optional[str] = None

    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "arg_name": self.arg_name,
            "arg_value": self.arg_value,
            "sources": self.sources,
            "tool_call_id": self.tool_call_id,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            tool_name=data["tool_name"],
            arg_name=data["arg_name"],
            arg_value=data["arg_value"],
            sources=data["sources"],
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class TraceData:
    file_path: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    dfg_nodes: List[DFGNode] = field(default_factory=list)
    system_prompt: str = ""
    user_prompt: str = ""
    success: bool = False

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "dfg_nodes": [dn.to_dict() for dn in self.dfg_nodes],
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            file_path=data["file_path"],
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            dfg_nodes=[DFGNode.from_dict(dn) for dn in data.get("dfg_nodes", [])],
            system_prompt=data.get("system_prompt", ""),
            user_prompt=data.get("user_prompt", ""),
            success=data.get("success", False),
        )


def _default_dict_set():
    """Helper for PolicyPart default factory"""
    from collections import defaultdict

    return defaultdict(set)


@dataclass
class PolicyPart:
    """Stores unique patterns (sets, not counts) for one category"""

    # Retrieval set → Action
    # Key: frozenset of retrieval tools seen in trace so far
    # Value: set of action tools called with this retrieval context
    cfg_ret_to_action: Dict[frozenset, set] = field(default_factory=_default_dict_set)

    # Action → Action (ALL pairs, not just consecutive)
    # Key: previous action tool
    # Value: set of action tools that appeared later in any trace
    cfg_action_to_action: Dict[str, set] = field(default_factory=_default_dict_set)

    # Action.arg → Source tools
    # Key: "action_tool.arg_name"
    # Value: set of retrieval tools (or user_prompt) that provided data
    dfg_arg_sources: Dict[str, set] = field(default_factory=_default_dict_set)
