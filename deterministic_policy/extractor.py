import json
import os
import requests
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict

from structures import TraceData, ToolCall, DFGNode, DFGSource


class BaseExtractor(ABC):
    def __init__(
        self,
        ignored_tools: Dict[str, Set[str]] = None,
        noisy_args: Dict[str, Set[str]] = None,
        tool_classification: Dict[str, Dict[str, Set[str]]] = None,
    ):
        self.ignored_tools = ignored_tools or {}
        self.noisy_args = noisy_args or {}
        # tool_classification[agent] = {"retrieval_tools": set(...), "action_tools": set(...)}
        self.tool_classification = tool_classification or {}

    def _get_retrieval_tools(self, agent_name: str) -> Set[str]:
        if not agent_name:
            return set()
        cls = self.tool_classification.get(agent_name, {})
        tools = cls.get("retrieval_tools")
        return set(tools) if tools else set()

    @abstractmethod
    def extract_data_flow(
        self,
        process_trace: TraceData,
        agent_name: str = None,
        target_tools: Set[str] = None,
        skip_nodes: Set[Tuple[str, str]] = None,  # (tool_call_id, arg_name)
    ) -> None:
        pass

    def _extract_strings(self, data: Any) -> Set[str]:
        """Recursively extract string leaf values from JSON-like structure"""
        strings = set()
        if isinstance(data, str):
            # Attempt to parse structure if it looks like one
            parsed = False
            s_stripped = data.strip()
            if s_stripped.startswith(("{", "[")):
                try:
                    # Try standard JSON first
                    obj = json.loads(data)
                    strings.update(self._extract_strings(obj))
                    parsed = True
                except:
                    try:
                        # Try replacing single quotes with double quotes (common in Python repr)
                        # We use a naive replacement which might break some strings but heuristics favor recall
                        obj = json.loads(data.replace("'", '"'))
                        strings.update(self._extract_strings(obj))
                        parsed = True
                    except:
                        pass

            if not parsed:
                strings.add(self._normalize(data))

        elif isinstance(data, (int, float, bool)):
            strings.add(self._normalize(str(data)))
        elif isinstance(data, dict):
            for v in data.values():
                strings.update(self._extract_strings(v))
        elif isinstance(data, list):
            for v in data:
                strings.update(self._extract_strings(v))
        return strings

    def _tokenize(self, text: str) -> Set[str]:
        return {self._normalize(text)}

    def _normalize(self, s: str) -> str:
        # Normalize to lowercase and remove non-alphanumeric characters
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _check_match(self, arg_strings: Set[str], source_strings: Set[str]) -> bool:
        """
        Return True if any arg string is contained in OR contains any source string.
        """
        for arg in arg_strings:
            if not arg:
                continue
            for src in source_strings:
                if not src:
                    continue
                # Exact match or containment
                if arg in src or src in arg:
                    return True
        return False


class TraceExtractor:
    def __init__(self, mode: str = "hybrid", debug: bool = False):
        self.mode = mode
        self.debug = debug
        self.ignored_tools: Dict[str, Set[str]] = {}
        self.noisy_args: Dict[str, Set[str]] = {}
        self.tool_classification: Dict[str, Dict[str, Set[str]]] = {}
        self._load_ignored_tools()

        self.extractors = []
        if mode == "heuristic":
            self.extractors.append(
                HeuristicExtractor(
                    self.ignored_tools,
                    self.noisy_args,
                    tool_classification=self.tool_classification,
                )
            )
        elif mode == "semantic":
            self.extractors.append(
                SemanticExtractor(
                    self.ignored_tools,
                    self.noisy_args,
                    tool_classification=self.tool_classification,
                )
            )
        elif mode == "hybrid":
            self.extractors.append(
                HeuristicExtractor(
                    self.ignored_tools,
                    self.noisy_args,
                    tool_classification=self.tool_classification,
                )
            )
            self.extractors.append(
                SemanticExtractor(
                    self.ignored_tools,
                    self.noisy_args,
                    tool_classification=self.tool_classification,
                )
            )
        else:
            raise ValueError(f"Unknown extraction mode: {mode}")

    def _load_ignored_tools(self):
        utils_dir = os.path.join(os.path.dirname(__file__), "utils")
        if os.path.exists(utils_dir):
            for agent_name in os.listdir(utils_dir):
                agent_dir = os.path.join(utils_dir, agent_name)
                if os.path.isdir(agent_dir):
                    # Load tool classification
                    class_path = os.path.join(agent_dir, "tool_classification.json")
                    if os.path.exists(class_path):
                        try:
                            with open(class_path, "r") as f:
                                cls_data = json.load(f)
                            self.tool_classification[agent_name] = {
                                "retrieval_tools": set(
                                    cls_data.get("retrieval_tools", [])
                                ),
                                "action_tools": set(cls_data.get("action_tools", [])),
                            }
                        except Exception as e:
                            print(
                                f"Error loading tool classification for {agent_name}: {e}"
                            )

                    # Load ignored tools
                    ignored_path = os.path.join(agent_dir, "ignored_tools.json")
                    if os.path.exists(ignored_path):
                        try:
                            with open(ignored_path, "r") as f:
                                tools = json.load(f)
                                self.ignored_tools[agent_name] = set(tools)
                        except Exception as e:
                            print(f"Error loading ignored tools for {agent_name}: {e}")

                    # Load noisy args
                    noisy_path = os.path.join(agent_dir, "noisy_args.json")
                    if os.path.exists(noisy_path):
                        try:
                            with open(noisy_path, "r") as f:
                                args = json.load(f)
                                # Initialize if not exists (though logic usually does this)
                                if agent_name not in self.noisy_args:
                                    self.noisy_args[agent_name] = set()
                                self.noisy_args[agent_name] = set(args)
                        except Exception as e:
                            print(f"Error loading noisy args for {agent_name}: {e}")

    def extract_from_file(
        self, file_path: str, agent_name: str = None, target_tools: Set[str] = None
    ) -> TraceData:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            if self.debug:
                print(f"Error reading file {file_path}: {e}")
            raise e

        trace = TraceData(file_path=file_path)
        trace.success = data.get("success", False)

        # 1. Parse Conversation History (Handle Utility vs Attack schemas)
        history = []
        # Priority 1: New Simulated Trace Schema (messages at root)
        if "messages" in data:
            history = data["messages"]
        # Priority 2: Old Simulated Trace Schema
        elif "simulated_trace" in data and "messages" in data["simulated_trace"]:
            history = data["simulated_trace"]["messages"]
        # Priority 3: Utility/Generic Schema
        elif "conversation_history" in data:
            history = data["conversation_history"]

        if not history and self.debug:
            print(f"Warning: No history found in {file_path}")

        self._parse_conversation(history, trace)

        # 2. Extract Data Flow (updates trace.dfg_nodes in place)
        if self.mode == "hybrid":
            # Hybrid Fallback Logic: Run heuristic first, then semantic only on unmatched
            heuristic = self.extractors[0]
            semantic = self.extractors[1]  # Assumes order [Heuristic, Semantic]

            # 1. Run Heuristic
            heuristic.debug = self.debug
            heuristic.extract_data_flow(trace, agent_name, target_tools=target_tools)

            # 2. Identify Matched Nodes
            matched_nodes = set()
            for node in trace.dfg_nodes:
                if node.sources:
                    matched_nodes.add((node.tool_call_id, node.arg_name))

            # 3. Run Semantic on Unmatched
            semantic.debug = self.debug
            try:
                semantic.extract_data_flow(
                    trace,
                    agent_name,
                    target_tools=target_tools,
                    skip_nodes=matched_nodes,
                )
            except Exception as e:
                # Hosted LLM calls can fail transiently (rate limits/timeouts).
                # Keep heuristic results so training/eval can proceed and caching still works.
                if self.debug:
                    print(
                        f"  [Semantic] Warning: semantic step failed; keeping heuristic only: {type(e).__name__}: {e}"
                    )

        else:
            for extractor in self.extractors:
                extractor.debug = self.debug
                try:
                    extractor.extract_data_flow(
                        trace, agent_name, target_tools=target_tools
                    )
                except Exception as e:
                    if self.debug:
                        print(
                            f"  [Extractor] Warning: extractor failed; continuing: {type(e).__name__}: {e}"
                        )

        # 3. Merge nodes if multiple extractors ran
        if len(self.extractors) > 1:
            self._merge_dfg_nodes(trace)

        # 4. Enforce provenance constraints: sources can only be retrieval tools or user_prompt.
        # This prevents nonsensical action-tool -> action-tool provenance edges.
        if agent_name:
            cls = self.tool_classification.get(agent_name, {})
            retrieval_tools = set(cls.get("retrieval_tools") or [])
            if retrieval_tools:
                for node in trace.dfg_nodes:
                    if not node.sources:
                        continue
                    node.sources = [
                        s
                        for s in node.sources
                        if (s.get("source_name") == "user_prompt")
                        or (s.get("source_name") in retrieval_tools)
                    ]

        return trace

    def _merge_dfg_nodes(self, trace: TraceData):
        """
        Merge DFGNodes that refer to the same (tool_call_id, arg_name).
        Combine their sources lists.
        """
        merged_map = {}  # Key: (tool_call_id, arg_name) -> DFGNode

        # Note: If tool_call_id is missing, we might have issues.
        # But we assume robust traces have IDs.
        # Fallback: We could key by (tool_name, arg_name, arg_value) but that's risky for duplicate calls.

        for node in trace.dfg_nodes:
            key = (node.tool_call_id, node.arg_name)

            if key not in merged_map:
                merged_map[key] = node
            else:
                # Merge sources
                existing_node = merged_map[key]

                # Create a set of existing signature tuples for deduplication
                existing_sigs = set()
                for src in existing_node.sources:
                    # DFGSource is stored as dict
                    sig = (src.get("source_name"), src.get("content"))
                    existing_sigs.add(sig)

                for src in node.sources:
                    sig = (src.get("source_name"), src.get("content"))
                    if sig not in existing_sigs:
                        existing_node.sources.append(src)
                        existing_sigs.add(sig)

        trace.dfg_nodes = list(merged_map.values())

    def _extract_text_content(self, content: Any) -> str:
        """
        Extract text content from various formats:
        - String: return as-is
        - List of dicts with 'content' key: concatenate text content
        - Other: convert to string
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle AgentDojo format: [{'type': 'text', 'content': '...'}]
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if "content" in item:
                        texts.append(str(item["content"]))
                    elif "text" in item:
                        texts.append(str(item["text"]))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts) if texts else str(content)
        elif isinstance(content, dict):
            if "content" in content:
                return str(content["content"])
            elif "text" in content:
                return str(content["text"])
            return str(content)
        return str(content) if content else ""

    def _parse_conversation(self, history: List[Dict], trace: TraceData):
        for msg in history:
            # Skip non-dict messages
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type", "")
            data = msg.get("data", {})
            content = data.get("content", "")

            if msg_type == "system":
                trace.system_prompt = content
            elif msg_type == "human":
                trace.user_prompt = content
            elif msg_type == "ai":
                tool_calls = data.get("additional_kwargs", {}).get("tool_calls", [])
                # Also handle direct tool_calls in data if present (some formats differ)
                if not tool_calls and "tool_calls" in data:
                    tool_calls = data["tool_calls"]

                for tc in tool_calls:
                    func = tc.get("function", {})
                    # function args are usually a json string
                    raw_args = func.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {"raw_args": raw_args}

                    trace.tool_calls.append(
                        ToolCall(
                            name=func.get("name", "unknown"),
                            arguments=args,
                            tool_call_id=tc.get("id"),
                            timestamp=None,  # Timestamp not always available in standard msg
                        )
                    )
            elif msg_type == "tool":
                # Associate output with the last tool call
                # Note: This assumes strict alternating order or sequential processing.
                # In robust systems, we should match by tool_call_id.
                # For this dataset, usually tool msg follows ai msg.
                tool_call_id = data.get("tool_call_id")
                # Find matching tool call in our list (reverse search is better)
                # But we constructed the list from AI messages just now.
                # Simplification: Attach to the last tool call that doesn't have output yet?
                # Or better: The AI message had tool_calls with IDs.
                # Let's retroactively fix this: we need IDs in ToolCall to match.

                # Re-iterating: The AI message loop processed all calls.
                # Now we see a tool output.
                # We need to find the tool call with this ID.
                # trace.tool_calls[-1].output = content # Naive assumption

                # Correct approach:
                # We need to store ID in ToolCall.
                pass

        # Second pass or smarter first pass required for matching outputs.
        # Let's do a more robust single pass state machine or ID mapping.
        trace.tool_calls = []  # Reset to do it properly

        pending_tool_calls = {}  # id -> ToolCall object
        msg_index = 0  # Track message index for generating synthetic IDs

        for msg in history:
            # Skip non-dict messages (some traces have strings or nulls)
            if not isinstance(msg, dict):
                continue

            msg_index += 1  # Increment for each valid message

            # Support both LangChain format (type) and OpenAI format (role)
            msg_type = msg.get("type", "")
            msg_role = msg.get("role", "")

            # Normalize: map OpenAI roles to LangChain types
            if not msg_type and msg_role:
                role_map = {
                    "system": "system",
                    "user": "human",
                    "assistant": "ai",
                    "tool": "tool",
                }
                msg_type = role_map.get(msg_role, "")

            # Content can be in different places depending on format
            data = msg.get("data", {})
            raw_content = data.get("content", "") or msg.get("content", "") or ""
            content = self._extract_text_content(raw_content)

            if msg_type == "system":
                trace.system_prompt = content
            elif msg_type == "human":
                trace.user_prompt = content
            elif msg_type == "ai":
                # Try multiple locations for tool_calls
                raw_tool_calls = data.get("additional_kwargs", {}).get("tool_calls", [])
                if not raw_tool_calls and "tool_calls" in data:
                    raw_tool_calls = data["tool_calls"]
                if not raw_tool_calls and "tool_calls" in msg:
                    raw_tool_calls = msg["tool_calls"] or []

                for tc_idx, tc in enumerate(raw_tool_calls):
                    tc_id = tc.get("id")

                    # CRITICAL FIX: Generate synthetic ID if missing
                    if not tc_id:
                        tc_id = f"call_{msg_index}_{tc_idx}"
                        if self.debug:
                            print(
                                f"  [DEBUG] Generated synthetic tool_call_id: {tc_id}"
                            )

                    func = tc.get("function", {})

                    # Handle case where 'function' is just the function name (string)
                    # This is the AgentDojo/generated_traces format
                    if isinstance(func, str):
                        func_name = func
                        raw_args = tc.get("args", {})
                        if isinstance(raw_args, dict):
                            args = raw_args
                        else:
                            try:
                                args = json.loads(raw_args or "{}")
                            except:
                                args = {}
                    # Standard OpenAI format where function is a dict
                    elif isinstance(func, dict):
                        func_name = func.get("name")
                        if not func_name:  # Try top level if 'function' wrapper missing
                            func_name = tc.get("name")
                            raw_args = tc.get("args")
                            if isinstance(raw_args, dict):
                                args = raw_args
                            else:
                                try:
                                    args = json.loads(raw_args or "{}")
                                except:
                                    args = {}
                        else:
                            raw_args = func.get("arguments", "{}")
                            try:
                                args = json.loads(raw_args)
                            except:
                                args = {}
                    else:
                        # Unknown format, skip
                        continue

                    tool_call = ToolCall(
                        name=func_name,
                        arguments=args,
                        tool_call_id=tc_id,  # Now guaranteed to have an ID
                    )
                    trace.tool_calls.append(tool_call)
                    pending_tool_calls[tc_id] = tool_call  # Always add to pending

            elif msg_type == "tool":
                # Try multiple locations for tool_call_id
                tc_id = data.get("tool_call_id") or msg.get("tool_call_id")
                if tc_id and tc_id in pending_tool_calls:
                    pending_tool_calls[tc_id].output = content
                elif not tc_id and len(trace.tool_calls) > 0:
                    # Fallback: Associate with most recent tool call that doesn't have output
                    for tool_call in reversed(trace.tool_calls):
                        if not tool_call.output:
                            tool_call.output = content
                            break


class HeuristicExtractor(BaseExtractor):
    """
    Enhanced heuristic-based data flow extractor with:
    - Format-aware tokenization (IBANs, emails, URLs, numbers, dates)
    - Multi-strategy matching (exact, substring, fuzzy, numeric proximity)
    - N-gram overlap for partial matches
    - Configurable thresholds
    """

    # Default configuration for matching thresholds
    DEFAULT_MATCH_CONFIG = {
        "fuzzy_threshold": 2,  # Max Levenshtein distance for fuzzy match
        "numeric_tolerance": 0.20,  # 20% tolerance for numeric proximity
        "ngram_size": 3,  # Character n-gram size
        "ngram_overlap_ratio": 0.5,  # Minimum n-gram overlap ratio
        "min_token_length_fuzzy": 5,  # Min length for fuzzy matching
        "min_token_length_ngram": 8,  # Min length for n-gram matching
    }

    # Regex patterns for format-aware extraction
    IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b", re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+')
    # Matches: 1234, 1,234.56, $1234, -1234.56, 1234.56%, etc.
    NUMBER_PATTERN = re.compile(r"[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+(?:\.\d+)?")
    # Matches various date formats
    DATE_PATTERNS = [
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # ISO: 2022-03-01
        re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),  # US: 03/01/2022
        re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b"),  # EU: 01.03.2022
        re.compile(r"\b\d{2}-\d{2}-\d{4}\b"),  # Alt: 01-03-2022
    ]
    # Phone number pattern (international formats)
    PHONE_PATTERN = re.compile(
        r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
    )

    # Common arithmetic transformations to detect
    ARITHMETIC_TRANSFORMATIONS = [
        ("half", lambda src, arg: abs(arg - src / 2) < 0.01),
        ("double", lambda src, arg: abs(arg - src * 2) < 0.01),
        ("tenth", lambda src, arg: abs(arg - src * 0.1) < 0.01),
        ("10% increase", lambda src, arg: abs(arg - src * 1.1) < 0.01),
        ("10% decrease", lambda src, arg: abs(arg - src * 0.9) < 0.01),
        ("15% increase", lambda src, arg: abs(arg - src * 1.15) < 0.01),
        ("20% increase", lambda src, arg: abs(arg - src * 1.2) < 0.01),
        ("quarter", lambda src, arg: abs(arg - src * 0.25) < 0.01),
        ("third", lambda src, arg: abs(arg - src / 3) < 0.1),
    ]

    def __init__(
        self,
        ignored_tools: Dict[str, Set[str]] = None,
        noisy_args: Dict[str, Set[str]] = None,
        tool_classification: Dict[str, Dict[str, Set[str]]] = None,
        match_config: Dict[str, Any] = None,
    ):
        super().__init__(
            ignored_tools, noisy_args, tool_classification=tool_classification
        )
        self.match_config = {**self.DEFAULT_MATCH_CONFIG, **(match_config or {})}
        self.debug = False

    def extract_data_flow(
        self,
        trace: TraceData,
        agent_name: str = None,
        target_tools: Set[str] = None,
        skip_nodes: Set[Tuple[str, str]] = None,
    ) -> None:
        """
        Enhanced heuristics for data flow extraction:
        1. Parse JSON outputs to strings
        2. Extract structured formats (IBANs, emails, numbers, dates)
        3. Multi-strategy matching with fallback chain
        """
        ignored_tools_set = (
            self.ignored_tools.get(agent_name, set()) if agent_name else set()
        )
        noisy_args_set = self.noisy_args.get(agent_name, set()) if agent_name else set()
        skip_nodes = skip_nodes or set()

        retrieval_tools_set = self._get_retrieval_tools(agent_name)

        processed_sources = []

        # Add User Prompt as first source
        user_prompt_data = self._prepare_source_data(
            trace.user_prompt, "user_prompt", -1
        )
        processed_sources.append(user_prompt_data)

        for i, tool_call in enumerate(trace.tool_calls):
            tool_ignored = tool_call.name in ignored_tools_set

            for arg_name, arg_val in tool_call.arguments.items():
                is_blacklisted = arg_name in noisy_args_set
                should_skip = tool_ignored or is_blacklisted

                # Check if this node should be skipped (already matched by previous extractor)
                node_key = (tool_call.tool_call_id, arg_name)
                if node_key in skip_nodes:
                    continue

                dfg_node = DFGNode(
                    tool_name=tool_call.name,
                    arg_name=arg_name,
                    arg_value=arg_val,
                    sources=[],
                    tool_call_id=tool_call.tool_call_id,
                )

                if should_skip:
                    continue

                # Prepare argument data for matching
                arg_data = self._prepare_arg_data(arg_val)

                # Compare against all previous sources
                for src in processed_sources:
                    if self._check_match_enhanced(arg_data, src):
                        dfg_node.sources.append(
                            {
                                "content": src["content"],
                                "source_name": src["type"],
                            }
                        )

                trace.dfg_nodes.append(dfg_node)

            # Add this tool's output to sources for future tools
            # IMPORTANT: only retrieval tools (and user_prompt) can be provenance sources.
            if tool_call.output and (tool_call.name in retrieval_tools_set):
                source_data = self._prepare_source_data(
                    tool_call.output, tool_call.name, i
                )
                processed_sources.append(source_data)

    def _prepare_source_data(
        self, content: str, source_type: str, index: int
    ) -> Dict[str, Any]:
        """Prepare comprehensive source data with all extracted formats."""
        return {
            "type": source_type,
            "content": content,
            "index": index,
            "tokens": self._extract_strings_enhanced(content),
            "ibans": self._extract_ibans(content),
            "emails": self._extract_emails(content),
            "urls": self._extract_urls(content),
            "numbers": self._extract_numbers(content),
            "dates": self._extract_dates(content),
            "phones": self._extract_phones(content),
            "ngrams": self._generate_ngrams(content),
            "words": self._tokenize_words(content),
        }

    def _prepare_arg_data(self, arg_val: Any) -> Dict[str, Any]:
        """Prepare argument data for matching."""
        # Convert to string representation for extraction
        if isinstance(arg_val, str):
            text = arg_val
        else:
            text = json.dumps(arg_val) if arg_val is not None else ""

        return {
            "raw": arg_val,
            "text": text,
            "tokens": self._extract_strings_enhanced(arg_val),
            "ibans": self._extract_ibans(text),
            "emails": self._extract_emails(text),
            "urls": self._extract_urls(text),
            "numbers": self._extract_numbers(text),
            "dates": self._extract_dates(text),
            "phones": self._extract_phones(text),
            "ngrams": self._generate_ngrams(text),
            "words": self._tokenize_words(text),
        }

    # ==================== Format-Aware Extraction ====================

    def _extract_ibans(self, text: str) -> Set[str]:
        """Extract IBAN-like strings (2 letters + 2 digits + up to 30 alphanumeric)."""
        if not isinstance(text, str):
            return set()
        matches = self.IBAN_PATTERN.findall(text)
        # Normalize: uppercase, no spaces/dashes
        return {re.sub(r"[\s-]", "", m.upper()) for m in matches}

    def _extract_emails(self, text: str) -> Set[str]:
        """Extract email addresses."""
        if not isinstance(text, str):
            return set()
        matches = self.EMAIL_PATTERN.findall(text)
        return {m.lower() for m in matches}

    def _extract_urls(self, text: str) -> Set[str]:
        """Extract URLs from text."""
        if not isinstance(text, str):
            return set()
        matches = self.URL_PATTERN.findall(text)
        # Normalize: lowercase, strip trailing punctuation
        return {m.lower().rstrip(".,;:") for m in matches}

    def _extract_numbers(self, text: str) -> Set[float]:
        """Extract numeric values, handling currency symbols and formatting."""
        if not isinstance(text, str):
            if isinstance(text, (int, float)):
                return {float(text)}
            return set()

        numbers = set()
        matches = self.NUMBER_PATTERN.findall(text)
        for m in matches:
            try:
                # Remove currency symbols, commas, percent signs
                cleaned = re.sub(r"[$,%]", "", m)
                cleaned = cleaned.replace(",", "")
                if cleaned and cleaned not in ("-", "+", "."):
                    numbers.add(float(cleaned))
            except ValueError:
                pass
        return numbers

    def _extract_dates(self, text: str) -> Set[str]:
        """Extract dates and normalize to ISO format (YYYY-MM-DD)."""
        if not isinstance(text, str):
            return set()

        dates = set()
        for pattern in self.DATE_PATTERNS:
            for match in pattern.findall(text):
                normalized = self._normalize_date(match)
                if normalized:
                    dates.add(normalized)
        return dates

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to ISO format."""
        try:
            # Try ISO format first
            if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                return date_str
            # US format: MM/DD/YYYY
            if "/" in date_str:
                parts = date_str.split("/")
                if len(parts) == 3:
                    return f"{parts[2]}-{parts[0]}-{parts[1]}"
            # EU format: DD.MM.YYYY or DD-MM-YYYY
            if "." in date_str or (
                date_str.count("-") == 2 and not date_str.startswith("20")
            ):
                sep = "." if "." in date_str else "-"
                parts = date_str.split(sep)
                if len(parts) == 3:
                    return f"{parts[2]}-{parts[1]}-{parts[0]}"
        except:
            pass
        return date_str

    def _extract_phones(self, text: str) -> Set[str]:
        """Extract phone numbers from text."""
        if not isinstance(text, str):
            return set()
        matches = self.PHONE_PATTERN.findall(text)
        # Normalize: remove all non-digit characters except leading +
        normalized = set()
        for m in matches:
            digits = re.sub(r"[^\d+]", "", m)
            if len(digits) >= 7:  # Minimum phone length
                normalized.add(digits)
        return normalized

    # ==================== String Extraction ====================

    def _extract_strings_enhanced(self, data: Any) -> Set[str]:
        """Recursively extract and normalize string values from data structures."""
        strings = set()
        if isinstance(data, str):
            parsed = False
            s_stripped = data.strip()
            if s_stripped.startswith(("{", "[")):
                try:
                    obj = json.loads(data)
                    strings.update(self._extract_strings_enhanced(obj))
                    parsed = True
                except:
                    try:
                        obj = json.loads(data.replace("'", '"'))
                        strings.update(self._extract_strings_enhanced(obj))
                        parsed = True
                    except:
                        pass

            if not parsed:
                # Add both normalized and canonicalized versions
                strings.add(self._normalize(data))
                strings.add(self._canonicalize(data))

        elif isinstance(data, (int, float)):
            strings.add(str(data))
        elif isinstance(data, bool):
            strings.add(str(data).lower())
        elif isinstance(data, dict):
            for v in data.values():
                strings.update(self._extract_strings_enhanced(v))
        elif isinstance(data, list):
            for v in data:
                strings.update(self._extract_strings_enhanced(v))

        # Filter out empty strings
        return {s for s in strings if s}

    def _canonicalize(self, s: str) -> str:
        """Normalize preserving some structure (spaces become single space)."""
        # Lowercase, collapse whitespace, keep alphanumeric and spaces
        normalized = re.sub(r"[^a-z0-9\s]", " ", s.lower())
        return " ".join(normalized.split())

    # ==================== Tokenization ====================

    def _tokenize(self, text: str) -> Set[str]:
        """Strict whole-string tokenization."""
        if not isinstance(text, str):
            return set()
        return {self._normalize(text)}

    def _tokenize_words(self, text: str) -> Set[str]:
        """Split text into normalized words for bag-of-words matching."""
        if not isinstance(text, str):
            return set()
        normalized = self._normalize_text(text)
        return {w for w in normalized.split() if w and len(w) > 1}

    def _normalize(self, s: str) -> str:
        """Strict normalization: lowercase, alphanumeric only."""
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _normalize_text(self, s: str) -> str:
        """Normalize but keep spaces for word splitting."""
        return re.sub(r"[^a-z0-9\s]", " ", s.lower())

    # ==================== N-Gram Generation ====================

    def _generate_ngrams(self, text: str, n: int = None) -> Set[str]:
        """Generate character-level n-grams for partial matching."""
        if not isinstance(text, str):
            return set()

        n = n or self.match_config["ngram_size"]
        # Normalize for n-gram generation
        normalized = self._normalize(text)

        if len(normalized) < n:
            return {normalized} if normalized else set()

        ngrams = set()
        for i in range(len(normalized) - n + 1):
            ngrams.add(normalized[i : i + n])
        return ngrams

    # ==================== Fuzzy Matching ====================

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _fuzzy_match(self, s1: str, s2: str) -> bool:
        """Check if two strings are within fuzzy threshold."""
        min_len = self.match_config["min_token_length_fuzzy"]
        threshold = self.match_config["fuzzy_threshold"]

        # Only apply fuzzy matching to strings of reasonable length
        if len(s1) < min_len or len(s2) < min_len:
            return False

        # Quick length check - if lengths differ too much, skip expensive calculation
        if abs(len(s1) - len(s2)) > threshold:
            return False

        distance = self._levenshtein_distance(s1, s2)
        return distance <= threshold

    # ==================== Numeric Proximity ====================

    def _numeric_proximity_match(
        self, arg_numbers: Set[float], source_numbers: Set[float]
    ) -> bool:
        """Check if any arg number is within tolerance of any source number."""
        confidence, _ = self._numeric_proximity_match_with_confidence(
            arg_numbers, source_numbers
        )
        return confidence > 0

    def _numeric_proximity_match_with_confidence(
        self, arg_numbers: Set[float], source_numbers: Set[float]
    ) -> Tuple[float, str]:
        """
        Check numeric proximity with confidence scoring.
        Returns (confidence, match_type) where:
        - confidence: 0.0-1.0 score
        - match_type: description of match (exact, proximity, transformation)
        """
        tolerance = self.match_config["numeric_tolerance"]

        for arg_num in arg_numbers:
            if arg_num == 0:
                continue
            for src_num in source_numbers:
                if src_num == 0:
                    continue

                # 1. Exact match (highest confidence)
                if abs(arg_num - src_num) < 0.001:
                    return (0.95, "exact")

                # 2. Arithmetic transformations (high confidence)
                for name, transform_fn in self.ARITHMETIC_TRANSFORMATIONS:
                    try:
                        if transform_fn(src_num, arg_num):
                            return (0.85, f"arithmetic:{name}")
                    except:
                        pass

                # 3. Percentage tolerance (medium confidence)
                ratio = arg_num / src_num
                if (1 - tolerance) <= ratio <= (1 + tolerance):
                    # Higher confidence for closer matches
                    closeness = 1 - abs(ratio - 1) / tolerance
                    confidence = 0.60 + 0.20 * closeness
                    return (confidence, f"proximity:{ratio:.2f}x")

                # 4. Small additive difference (lower confidence)
                if abs(arg_num - src_num) <= 10:
                    confidence = 0.50 - abs(arg_num - src_num) * 0.03
                    return (max(0.30, confidence), "additive")

        return (0.0, "none")

    # ==================== N-Gram Overlap ====================

    def _ngram_overlap_match(
        self, arg_ngrams: Set[str], source_ngrams: Set[str]
    ) -> bool:
        """Check if n-gram overlap exceeds threshold."""
        min_len = self.match_config["min_token_length_ngram"]
        overlap_ratio = self.match_config["ngram_overlap_ratio"]

        if not arg_ngrams or not source_ngrams:
            return False

        # Only apply to sufficiently long content
        if len(arg_ngrams) < min_len // self.match_config["ngram_size"]:
            return False

        intersection = arg_ngrams & source_ngrams
        # Ratio based on smaller set (the argument)
        ratio = len(intersection) / len(arg_ngrams) if arg_ngrams else 0

        return ratio >= overlap_ratio

    # ==================== Main Matching Logic ====================

    def _check_match_enhanced(
        self, arg_data: Dict[str, Any], source: Dict[str, Any]
    ) -> bool:
        """
        Multi-strategy matching with fallback chain:
        1. Format-specific matching (IBANs, emails, URLs, dates)
        2. Exact/substring containment
        3. Numeric proximity
        4. Word overlap (bag-of-words)
        5. Fuzzy matching (edit distance)
        6. N-gram overlap
        """

        # Strategy 1: Format-specific exact matches (highest confidence)
        # IBANs
        if arg_data["ibans"] and source["ibans"]:
            if arg_data["ibans"] & source["ibans"]:
                return True

        # Emails
        if arg_data["emails"] and source["emails"]:
            if arg_data["emails"] & source["emails"]:
                return True

        # URLs
        if arg_data["urls"] and source["urls"]:
            if arg_data["urls"] & source["urls"]:
                return True

        # Dates
        if arg_data["dates"] and source["dates"]:
            if arg_data["dates"] & source["dates"]:
                return True

        # Phones
        if arg_data["phones"] and source["phones"]:
            if arg_data["phones"] & source["phones"]:
                return True

        # Strategy 2: Exact/substring containment on normalized tokens
        arg_tokens = arg_data["tokens"]
        source_tokens = source["tokens"]

        for arg in arg_tokens:
            if not arg or len(arg) < 2:
                continue
            for src in source_tokens:
                if not src:
                    continue
                # Exact match
                if arg == src:
                    return True
                # Substring containment (only for reasonably sized strings)
                if len(arg) >= 4 and len(src) >= 4:
                    if arg in src or src in arg:
                        return True

        # Strategy 3: Numeric proximity
        if arg_data["numbers"] and source["numbers"]:
            # First check exact number match
            if arg_data["numbers"] & source["numbers"]:
                return True
            # Then check proximity
            if self._numeric_proximity_match(arg_data["numbers"], source["numbers"]):
                return True

        # Strategy 4: Word overlap (all arg words in source)
        arg_words = arg_data["words"]
        source_words = source["words"]

        if arg_words and len(arg_words) >= 2:
            # All meaningful arg words should appear in source
            if arg_words.issubset(source_words):
                return True

        # Strategy 5: Fuzzy matching for normalized tokens
        for arg in arg_tokens:
            if not arg:
                continue
            for src in source_tokens:
                if not src:
                    continue
                if self._fuzzy_match(arg, src):
                    return True

        # Strategy 6: N-gram overlap (for partial/corrupted matches)
        if self._ngram_overlap_match(arg_data["ngrams"], source["ngrams"]):
            return True

        return False


class SemanticExtractor(BaseExtractor):
    def __init__(
        self,
        ignored_tools: Dict[str, Set[str]] = None,
        noisy_args: Dict[str, Set[str]] = None,
        tool_classification: Dict[str, Dict[str, Set[str]]] = None,
    ):
        super().__init__(
            ignored_tools, noisy_args, tool_classification=tool_classification
        )

        # Get LLM configuration from environment variables
        self.llm_provider = os.environ.get("EXTRACTOR_LLM_PROVIDER", "ollama").lower()
        self.llm_model = os.environ.get(
            "EXTRACTOR_LLM_MODEL", os.environ.get("QWEN_MODEL_NAME", "llama3.3:70b")
        )
        self.llm_base_url = os.environ.get(
            "EXTRACTOR_LLM_BASE_URL",
            os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        self.api_key = os.environ.get(
            "EXTRACTOR_API_KEY", os.environ.get("DEEPINFRA_API_KEY")
        )

        if self.llm_provider == "ollama":
            self._warmup_model()

    def _warmup_model(self):
        print(f"  [Semantic] Warming up model {self.llm_model}...")
        try:
            requests.post(
                f"{self.llm_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": "warmup",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=300,
            )
            print("  [Semantic] Model warmed up.")
        except Exception as e:
            print(f"  [Semantic] Warning: Model warmup failed: {e}")

    def _call_llm(self, prompt: str) -> Any:
        """Call LLM based on configured provider."""
        if self.llm_provider == "ollama":
            return self._call_ollama(prompt)
        elif self.llm_provider == "deepinfra":
            return self._call_deepinfra(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def _call_ollama(self, prompt: str, retry_count: int = 3) -> Any:
        """Call Ollama API with retry logic for 500 errors."""
        last_error = None
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    f"{self.llm_base_url}/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 512,  # Limit response length
                            "num_ctx": 8192,  # Context window
                        },
                    },
                    timeout=300,
                )
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                if "response" in result:
                    return json.loads(result["response"])
                else:
                    print(f"  [Semantic] Warning: Unexpected response format: {result}")
                    return {"sources": []}
                    
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response.status_code == 500:
                    print(f"  [Semantic] Retry {attempt + 1}/{retry_count} due to 500 error")
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    raise
            except json.JSONDecodeError as e:
                print(f"  [Semantic] JSON decode error: {e}")
                return {"sources": []}
            except Exception as e:
                print(f"  [Semantic] Unexpected error: {e}")
                raise
        
        # All retries failed
        print(f"  [Semantic] All retries failed: {last_error}")
        return {"sources": []}

        self.api_key = os.environ.get(
            "EXTRACTOR_API_KEY", os.environ.get("DEEPINFRA_API_KEY")
        )

    def _call_deepinfra(self, prompt: str) -> Any:
        """Call DeepInfra API for semantic extraction."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            self.llm_base_url,
            headers=headers,
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=300,
        )
        response.raise_for_status()
        try:
            content = response.json()["choices"][0]["message"]["content"]
            # Clean up markdown code blocks if present
            if "```" in content:
                content = content.replace("```json", "").replace("```", "").strip()
            print(f"  [Semantic] Raw content: {content}")
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback 1: try to find specific {"sources": [...]} pattern
            # Matches {"sources": [0, 1]} allowing for whitespace
            match = re.search(r'\{"sources":\s*\[[^\]]*\]\}', content)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass

            # Fallback 2: Try to handle \boxed{...} or other artifacts by finding ALL {...}
            try:
                # Find all {...} candidates
                candidates = re.findall(r"\{.*?\}", content, re.DOTALL)
                for cand in reversed(candidates):
                    try:
                        obj = json.loads(cand)
                        if "sources" in obj:
                            return obj
                    except:
                        continue
            except:
                pass

            print(f"  [Semantic] Error decoding JSON from DeepInfra.")
            print(f"  [Semantic] Raw content: {content}")
            raise
        except Exception as e:
            raise e
        except Exception as e:
            raise e

    def extract_data_flow(
        self,
        trace: TraceData,
        agent_name: str = None,
        target_tools: Set[str] = None,
        skip_nodes: Set[Tuple[str, str]] = None,
    ) -> None:
        """
        Semantic data flow extraction using LLM.
        """
        ignored_tools_set = (
            self.ignored_tools.get(agent_name, set()) if agent_name else set()
        )
        noisy_args_set = self.noisy_args.get(agent_name, set()) if agent_name else set()
        skip_nodes = skip_nodes or set()

        retrieval_tools_set = self._get_retrieval_tools(agent_name)

        sources_list = []
        sources_list.append(
            {"type": "user_prompt", "content": trace.user_prompt, "id": 0}
        )

        # sources_list indices: 0 is user_prompt. 1..N are tools.
        # We need to maintain this index mapping.

        for i, tool_call in enumerate(trace.tool_calls):
            # i corresponds to tool index. Source index will be i+1 (becaus 0 is user prompt)
            # But wait, for the *current* tool call, available sources are indices 0 to i.

            # Prepare candidates
            candidates = sources_list  # All past sources

            # Identify which args need analysis
            args_to_analyze = {}
            tool_ignored = tool_call.name in ignored_tools_set
            if target_tools is not None and tool_call.name not in target_tools:
                tool_ignored = True

            nodes_for_this_call = []

            for arg_name, arg_val in tool_call.arguments.items():
                is_blacklisted = arg_name in noisy_args_set
                should_skip = tool_ignored or is_blacklisted

                # Check if this node should be skipped (already matched by previous extractor)
                node_key = (tool_call.tool_call_id, arg_name)
                if node_key in skip_nodes:
                    continue

                if should_skip:
                    continue

                node = DFGNode(
                    tool_name=tool_call.name,
                    arg_name=arg_name,
                    arg_value=arg_val,
                    sources=[],
                    tool_call_id=tool_call.tool_call_id,
                )
                nodes_for_this_call.append(node)
                args_to_analyze[arg_name] = arg_val

            if not candidates:
                trace.dfg_nodes.extend(nodes_for_this_call)
            else:
                for node in nodes_for_this_call:
                    if node.arg_name in args_to_analyze:
                        # Construct single arg prompt
                        prompt = self._construct_single_arg_prompt(
                            tool_call.name, node.arg_name, node.arg_value, candidates
                        )

                        # Call LLM
                        # print(
                        #     f"  [Semantic] Analyzing {tool_call.name}.{node.arg_name} value='{node.arg_value}'"
                        # )
                        response_data = self._call_llm(prompt)
                        # print(f"    [Semantic] Raw response: {response_data}")

                        # Robustness: Prompt now asks for {"sources": [...]}, but handle variations
                        raw_indices = []

                        if isinstance(response_data, dict):
                            # Best case
                            if "sources" in response_data:
                                raw_indices = response_data["sources"]
                            # Fallback: model made up a key like "source_index" or "indices"
                            elif "source_index" in response_data:
                                raw_indices = response_data["source_index"]
                            elif "indices" in response_data:
                                raw_indices = response_data["indices"]
                            else:
                                # Try to find ANY list value
                                for v in response_data.values():
                                    if isinstance(v, list):
                                        raw_indices = v
                                        break
                                    elif isinstance(v, int):
                                        raw_indices = [v]
                                        break
                        elif isinstance(response_data, list):
                            # Model returned bare list despite instructions (it happens)
                            raw_indices = response_data
                        elif isinstance(response_data, int):
                            raw_indices = [response_data]

                        if isinstance(raw_indices, int):
                            raw_indices = [raw_indices]

                        if isinstance(raw_indices, list):
                            if raw_indices:
                                # print(f"    [Semantic] Parsed indices: {raw_indices}")
                                pass
                            for idx in raw_indices:
                                if isinstance(idx, int) and 0 <= idx < len(candidates):
                                    src = candidates[idx]

                                    # --- VERBATIM CHECK ---
                                    # For sensitive args AND specific identifiers, enforce that semantic match must also satisfy containment
                                    # This prevents "hallucinating" data flow from causal relationships
                                    verbatim_args = {
                                        "password",
                                        "iban",
                                        "file_path",
                                        "email",
                                        "username",
                                        "id",
                                        "recipient",
                                        "recipients",
                                        "hotel_names",
                                        "restaurant_names",
                                        "company_name",
                                        "query",
                                        "subject",
                                        "event_id",
                                        "participants",
                                        "new_start_time",
                                        "new_end_time",
                                        "date",
                                    }
                                    if node.arg_name in verbatim_args:
                                        # Only extract source tokens if needed (lazy optimization?)
                                        # Extract strings from source content
                                        source_tokens = set()
                                        try:
                                            # Try JSON parse first
                                            loaded = json.loads(src["content"])
                                            source_tokens = self._extract_strings(
                                                loaded
                                            )
                                        except:
                                            source_tokens = self._tokenize(
                                                src["content"]
                                            )

                                        arg_tokens = self._extract_strings(
                                            node.arg_value
                                        )

                                        if not self._check_match(
                                            arg_tokens, source_tokens
                                        ):
                                            continue

                                    node.sources.append(
                                        {
                                            "content": src["content"],
                                            "source_name": src["type"],
                                        }
                                    )
                                else:
                                    # print(
                                    #     f"    [Semantic] Index {idx} out of bounds (0-{len(candidates)-1}) or invalid."
                                    # )
                                    pass
                        else:
                            # print(
                            #     f"    [Semantic] Unexpected response for {node.arg_name}: {raw_indices}"
                            # )
                            pass

                trace.dfg_nodes.extend(nodes_for_this_call)

            # Add this tool's output to sources for future tools
            # IMPORTANT: only retrieval tools (and user_prompt) can be provenance sources.
            if tool_call.output and (tool_call.name in retrieval_tools_set):
                sources_list.append(
                    {
                        "type": tool_call.name,
                        "content": tool_call.output,
                        "id": len(sources_list),
                    }
                )

    def _construct_single_arg_prompt(
        self, tool_name: str, arg_name: str, arg_val: Any, candidates: List[Dict]
    ) -> str:
        # Truncate very long content to avoid context overflow
        max_content_length = 1000
        
        prompt = f"""CRITICAL: You MUST systematically check EVERY source (0, 1, 2, 3...), NOT just Source_0!

Task: Find which source(s) contain the value '{arg_val}' for argument '{arg_name}' in tool '{tool_name}'.

Argument to find: '{arg_name}' = '{arg_val}'

Candidate Sources (check ALL of them):
"""
        
        for idx, src in enumerate(candidates):
            content = str(src["content"])
            # Truncate very long content but keep it readable
            if len(content) > max_content_length:
                content = content[:max_content_length] + "... [truncated]"
            prompt += f"\n[Source_{idx}] ({src['type']}):\n{content}\n"

        prompt += f"""

STEP-BY-STEP PROCESS:
1. First, search for '{arg_val}' in Source_1, Source_2, Source_3, etc. (tool outputs, files)
2. Then check Source_0 (user_prompt) 
3. List ALL sources that contain this value

MATCHING RULES:
- Exact match: value appears word-for-word in the source
- Substring match: value is part of a larger string in the source  
- Format variation: "$100" matches "100", "UK123..." matches "UK123..."
- Inference: value is clearly derived from the source (e.g., amount from a bill file)

IMPORTANT EXAMPLES:

Example 1 (Value in Source_1, NOT Source_0):
Argument: recipient = 'UK12345678901234567890'
Source_0: "Pay the bill from file.txt"
Source_1: "Bill details: IBAN UK12345678901234567890, Amount: $100"
Correct answer: {{"sources": [1]}}  <- Because value is in Source_1!
Wrong answer: {{"sources": [0]}}   <- This is WRONG! Source_0 doesn't contain the IBAN!

Example 2 (Value in multiple sources):
Argument: amount = '150'
Source_0: "Transfer $150 to my friend"
Source_1: "Balance: 150.00"
Correct answer: {{"sources": [0, 1]}}  <- Both contain it

Example 3 (Value only in Source_0):
Argument: subject = 'Rent payment'
Source_0: "Send rent payment to landlord"
Source_1: "Account balance: $1000"
Correct answer: {{"sources": [0]}}  <- Only in Source_0

Example 4 (No match):
Argument: id = '12345'
Source_0: "Get my recent transactions"
Correct answer: {{"sources": []}}  <- Value not found anywhere

Now analyze where '{arg_val}' appears. Return ONLY valid JSON:
{{"sources": [list of matching source indices]}}
"""
        return prompt
