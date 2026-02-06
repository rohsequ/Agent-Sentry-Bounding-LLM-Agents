"""
Three-Part Set-Based Policy with Fail-Secure Scoring
No backward compatibility with previous versions.
"""

import json
import ast
from pathlib import Path
from typing import List, Set, Dict, Tuple, Any
from structures import TraceData, DFGNode, PolicyPart


class ActionBlockingPolicy:
    """Three-part set-based policy with fail-secure scoring"""

    def __init__(self):
        # Three-part pattern storage
        self.utility = PolicyPart()
        self.ambiguous = PolicyPart()
        self.attack = PolicyPart()

        # Tool classifications
        self.action_tools: Set[str] = set()
        self.retrieval_tools: Set[str] = set()

    def load_tool_classifications(self, agent_name: str):
        """Load retrieval and action tool classifications"""
        path = Path(__file__).parent / "utils" / agent_name / "tool_classification.json"
        with open(path) as f:
            data = json.load(f)

        self.retrieval_tools = set(data.get("retrieval_tools", []))
        self.action_tools = set(data.get("action_tools", []))

        print(f"Loaded tool classifications for {agent_name}:")
        print(f"  Retrieval tools: {len(self.retrieval_tools)}")
        print(f"  Action tools: {len(self.action_tools)}")

    def train_utility(self, utility_traces: List[TraceData]):
        """Phase 1: Extract unique patterns from benign traces"""

        print(
            f"\nPhase 1: Extracting patterns from {len(utility_traces)} utility traces..."
        )

        for trace in utility_traces:
            tool_seq = ["user_prompt"] + [tc.name for tc in trace.tool_calls]

            # === CFG: Accumulate context ===
            ret_tools_seen = set()  # Accumulate retrieval tools
            actions_seen = []  # Accumulate all actions

            for tool in tool_seq:
                if tool == "user_prompt":
                    continue

                if tool in self.retrieval_tools:
                    ret_tools_seen.add(tool)

                elif tool in self.action_tools:
                    # Record: retrieval set → action
                    ret_key = frozenset(ret_tools_seen)
                    self.utility.cfg_ret_to_action[ret_key].add(tool)

                    # Record: ALL previous actions → current action
                    for prev_action in actions_seen:
                        self.utility.cfg_action_to_action[prev_action].add(tool)

                    actions_seen.append(tool)

            # === DFG: Argument sources ===
            for node in trace.dfg_nodes:
                if node.tool_name in self.action_tools:
                    arg_key = f"{node.tool_name}.{node.arg_name}"

                    for src in node.sources:
                        src_name = src.get("source_name")
                        if (
                            src_name == "user_prompt"
                            or src_name in self.retrieval_tools
                        ):
                            self.utility.dfg_arg_sources[arg_key].add(src_name)

        print(f"✓ Utility patterns extracted:")
        print(f"  CFG ret→action: {len(self.utility.cfg_ret_to_action)}")
        print(f"  CFG action→action: {len(self.utility.cfg_action_to_action)}")
        print(f"  DFG arg sources: {len(self.utility.dfg_arg_sources)}")

    def train_attack(self, attack_traces: List[TraceData]):
        """Phase 2: Extract attack patterns and partition into ambiguous/attack-only"""

        print(
            f"\nPhase 2: Extracting patterns from {len(attack_traces)} attack traces..."
        )

        # Temporary storage for attack patterns
        attack_temp = PolicyPart()

        # Extract patterns from attack traces (same logic as Phase 1)
        for trace in attack_traces:
            tool_seq = ["user_prompt"] + [tc.name for tc in trace.tool_calls]
            ret_tools_seen = set()
            actions_seen = []

            for tool in tool_seq:
                if tool == "user_prompt":
                    continue

                if tool in self.retrieval_tools:
                    ret_tools_seen.add(tool)

                elif tool in self.action_tools:
                    ret_key = frozenset(ret_tools_seen)
                    attack_temp.cfg_ret_to_action[ret_key].add(tool)

                    for prev_action in actions_seen:
                        attack_temp.cfg_action_to_action[prev_action].add(tool)

                    actions_seen.append(tool)

            for node in trace.dfg_nodes:
                if node.tool_name in self.action_tools:
                    arg_key = f"{node.tool_name}.{node.arg_name}"
                    for src in node.sources:
                        src_name = src.get("source_name")
                        if (
                            src_name == "user_prompt"
                            or src_name in self.retrieval_tools
                        ):
                            attack_temp.dfg_arg_sources[arg_key].add(src_name)

        print(f"✓ Attack patterns extracted, partitioning...")

        # === PARTITIONING LOGIC ===

        # CFG: Retrieval → Action
        for ret_set, actions in attack_temp.cfg_ret_to_action.items():
            for action in actions:
                if (
                    ret_set in self.utility.cfg_ret_to_action
                    and action in self.utility.cfg_ret_to_action[ret_set]
                ):
                    # Pattern in BOTH → move to ambiguous
                    self.ambiguous.cfg_ret_to_action[ret_set].add(action)
                    self.utility.cfg_ret_to_action[ret_set].discard(action)
                else:
                    # Pattern ONLY in attack → attack part
                    self.attack.cfg_ret_to_action[ret_set].add(action)

        # CFG: Action → Action
        for prev_action, next_actions in attack_temp.cfg_action_to_action.items():
            for next_action in next_actions:
                if (
                    prev_action in self.utility.cfg_action_to_action
                    and next_action in self.utility.cfg_action_to_action[prev_action]
                ):
                    # Ambiguous
                    self.ambiguous.cfg_action_to_action[prev_action].add(next_action)
                    self.utility.cfg_action_to_action[prev_action].discard(next_action)
                else:
                    # Attack-only
                    self.attack.cfg_action_to_action[prev_action].add(next_action)

        # DFG: Action.arg → Sources
        # IMPORTANT: Keep ALL sources together for each arg_key
        # Classify based on whether ANY source is attack-only
        # DFG: Action.arg → Sources
        # IMPORTANT: Keep ALL sources together for each arg_key
        # Classify based on whether ANY source is attack-only
        for arg_key, attack_sources in attack_temp.dfg_arg_sources.items():
            if arg_key in self.utility.dfg_arg_sources:
                utility_sources = self.utility.dfg_arg_sources[arg_key]

                # Intersection -> Ambiguous (seen in both)
                shared = attack_sources & utility_sources
                if shared:
                    current_ambiguous = self.ambiguous.dfg_arg_sources.get(
                        arg_key, set()
                    )
                    self.ambiguous.dfg_arg_sources[arg_key] = current_ambiguous | shared
                    # Remove shared from utility
                    self.utility.dfg_arg_sources[arg_key] = utility_sources - shared

                # Unique to Attack -> Attack
                unique_attack = attack_sources - utility_sources
                if unique_attack:
                    self.attack.dfg_arg_sources[arg_key] = unique_attack
            else:
                # Key only in attack -> Attack-only
                self.attack.dfg_arg_sources[arg_key] = attack_sources

        # Clean up empty entries
        self._remove_empty_entries()

        print(f"✓ Partitioning complete:")
        print(
            f"  Utility: {len(self.utility.cfg_ret_to_action)} CFG ret→action, "
            f"{len(self.utility.cfg_action_to_action)} action→action, "
            f"{len(self.utility.dfg_arg_sources)} DFG"
        )
        print(
            f"  Ambiguous: {len(self.ambiguous.cfg_ret_to_action)} CFG ret→action, "
            f"{len(self.ambiguous.cfg_action_to_action)} action→action, "
            f"{len(self.ambiguous.dfg_arg_sources)} DFG"
        )
        print(
            f"  Attack: {len(self.attack.cfg_ret_to_action)} CFG ret→action, "
            f"{len(self.attack.cfg_action_to_action)} action→action, "
            f"{len(self.attack.dfg_arg_sources)} DFG"
        )

    def _remove_empty_entries(self):
        """Remove empty dict entries after partitioning"""
        for part in [self.utility, self.ambiguous, self.attack]:
            empty_keys = [k for k, v in part.cfg_ret_to_action.items() if not v]
            for k in empty_keys:
                del part.cfg_ret_to_action[k]

            empty_keys = [k for k, v in part.cfg_action_to_action.items() if not v]
            for k in empty_keys:
                del part.cfg_action_to_action[k]

            empty_keys = [k for k, v in part.dfg_arg_sources.items() if not v]
            for k in empty_keys:
                del part.dfg_arg_sources[k]

    def train(self, utility_traces: List[TraceData], attack_traces: List[TraceData]):
        """
        Two-phase training - NO backward compatibility, both required
        """
        if not attack_traces:
            raise ValueError("attack_traces is required for training")

        self.train_utility(utility_traces)
        self.train_attack(attack_traces)
        print(f"\n✓ Policy training complete!")

    def _classify_pattern(self, pattern_key, value, util_dict, amb_dict, att_dict):
        """
        Classify a pattern: "attack", "ambiguous", "utility", or "novel"
        Check in fail-secure order: attack → ambiguous → utility → novel
        """
        if pattern_key in att_dict and value in att_dict[pattern_key]:
            return "attack"
        elif pattern_key in amb_dict and value in amb_dict[pattern_key]:
            return "ambiguous"
        elif pattern_key in util_dict and value in util_dict[pattern_key]:
            return "utility"
        else:
            return "novel"

    def _score_cfg(self, action_name, actions_seen_so_far, ret_tools_seen_so_far):
        """
        Score CFG with fail-secure priority:
        - If ANY component is "attack" → +10
        - Else if ANY is "ambiguous" → 0
        - Else if ALL are "utility" → -10
        - Else (novel) → +3
        """
        component_classes = []

        # Component 1: Retrieval → Action
        ret_key = frozenset(ret_tools_seen_so_far)
        ret_class = self._classify_pattern(
            ret_key,
            action_name,
            self.utility.cfg_ret_to_action,
            self.ambiguous.cfg_ret_to_action,
            self.attack.cfg_ret_to_action,
        )
        component_classes.append(ret_class)

        # Component 2: Action → Action (check ALL previous actions)
        if actions_seen_so_far:
            action_class = "novel"

            for prev_action in actions_seen_so_far:
                cls = self._classify_pattern(
                    prev_action,
                    action_name,
                    self.utility.cfg_action_to_action,
                    self.ambiguous.cfg_action_to_action,
                    self.attack.cfg_action_to_action,
                )

                # Priority: attack > ambiguous > utility > novel
                if cls == "attack":
                    action_class = "attack"
                    break  # Stop immediately
                elif cls == "ambiguous" and action_class != "attack":
                    action_class = "ambiguous"
                elif cls == "utility" and action_class not in ["attack", "ambiguous"]:
                    action_class = "utility"

            component_classes.append(action_class)

        # Apply fail-secure priority logic across ALL components
        if "attack" in component_classes:
            return +10.0, "attack"
        elif "ambiguous" in component_classes:
            return 0.0, "ambiguous"
        elif all(c == "utility" for c in component_classes):
            return -10.0, "utility"
        else:
            return +3.0, "novel"

    def _score_dfg(self, action_name, dfg_nodes):
        """
        Score DFG with fail-secure priority:
        - If ANY source is "attack" → +5
        - Else if ANY is "ambiguous" → 0
        - Else if ALL are "utility" → -5
        - Else (novel) → +2
        """
        source_classes = []

        for node in dfg_nodes:
            if node.tool_name == action_name:
                arg_key = f"{action_name}.{node.arg_name}"

                for src in node.sources:
                    src_name = src.get("source_name")

                    if (
                        src_name != "user_prompt"
                        and src_name not in self.retrieval_tools
                    ):
                        continue

                    cls = self._classify_pattern(
                        arg_key,
                        src_name,
                        self.utility.dfg_arg_sources,
                        self.ambiguous.dfg_arg_sources,
                        self.attack.dfg_arg_sources,
                    )
                    source_classes.append(cls)

        # Apply fail-secure priority
        if "attack" in source_classes:
            return +5.0, "attack"
        elif "ambiguous" in source_classes:
            return 0.0, "ambiguous"
        elif source_classes and all(c == "utility" for c in source_classes):
            return -5.0, "utility"
        elif not source_classes:
            return 0.0, "ambiguous"
        else:
            return +2.0, "novel"

    def score_action(
        self,
        action_name,
        actions_seen_so_far,
        ret_tools_seen_so_far,
        dfg_nodes,
        detailed=False,
    ):
        """
        Score an action before execution.

        Returns:
            float: Total score (negative = safe, positive = suspicious)
            or dict if detailed=True
        """
        cfg_score, cfg_class = self._score_cfg(
            action_name, actions_seen_so_far, ret_tools_seen_so_far
        )

        dfg_score, dfg_class = self._score_dfg(action_name, dfg_nodes)

        total_score = cfg_score + dfg_score

        if detailed:
            return {
                "score": total_score,
                "cfg_score": cfg_score,
                "cfg_class": cfg_class,
                "dfg_score": dfg_score,
                "dfg_class": dfg_class,
            }

        return total_score

    def should_block_action(
        self,
        action_name,
        actions_seen_so_far,
        ret_tools_seen_so_far,
        dfg_nodes,
        threshold=0.0,
    ):
        """Determine if action should be blocked (score > threshold)"""
        result = self.score_action(
            action_name,
            actions_seen_so_far,
            ret_tools_seen_so_far,
            dfg_nodes,
            detailed=True,
        )

        return result["score"] > threshold, result["score"], result

    def save(self, path: str):
        """Save policy to JSON"""

        def serialize_part(part: PolicyPart) -> dict:
            return {
                "cfg_ret_to_action": {
                    str(sorted(k)): list(v) for k, v in part.cfg_ret_to_action.items()
                },
                "cfg_action_to_action": {
                    k: list(v) for k, v in part.cfg_action_to_action.items()
                },
                "dfg_arg_sources": {
                    k: list(v) for k, v in part.dfg_arg_sources.items()
                },
            }

        data = {
            "version": "3.0",
            "utility": serialize_part(self.utility),
            "ambiguous": serialize_part(self.ambiguous),
            "attack": serialize_part(self.attack),
            "action_tools": list(self.action_tools),
            "retrieval_tools": list(self.retrieval_tools),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ActionBlockingPolicy":
        """Load policy from JSON"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("version") != "3.0":
            raise ValueError(
                f"Incompatible version: {data.get('version')}. Expected 3.0"
            )

        policy = cls()

        def deserialize_part(part_data: dict) -> PolicyPart:
            part = PolicyPart()

            for k_str, v_list in part_data.get("cfg_ret_to_action", {}).items():
                key = frozenset(ast.literal_eval(k_str))
                part.cfg_ret_to_action[key] = set(v_list)

            for k, v_list in part_data.get("cfg_action_to_action", {}).items():
                part.cfg_action_to_action[k] = set(v_list)

            for k, v_list in part_data.get("dfg_arg_sources", {}).items():
                part.dfg_arg_sources[k] = set(v_list)

            return part

        policy.utility = deserialize_part(data["utility"])
        policy.ambiguous = deserialize_part(data["ambiguous"])
        policy.attack = deserialize_part(data["attack"])

        policy.action_tools = set(data.get("action_tools", []))
        policy.retrieval_tools = set(data.get("retrieval_tools", []))

        return policy
