#!/usr/bin/env python3
"""
Evaluate trained policies on AgentDojo dataset.

For utility: Uses none.json files under user_task_<> folders where utility=True
For attacks: Uses json files under user_task_<>/important_instructions/ where security=True
"""
import json
import argparse
import os
from pathlib import Path
from collections import defaultdict
import sys
from typing import Optional

# Add parent directory to path to import policy module
sys.path.insert(0, str(Path(__file__).parent))
from policy import ActionBlockingPolicy
from extractor import TraceExtractor

# Intent alignment results path (workspace root is parent of deterministic_policy)
BASE_DIR = Path(__file__).resolve().parent.parent
INTENT_ALIGNMENT_ROOT = str(BASE_DIR / "agentdojo_intent_alignment_results")


def load_intent_alignment_result(
    file_path: str,
    agent_name: str,
    is_attack: bool,
    model_name: str = "gpt-5-nano",
    intent_root: Optional[str] = None,
) -> Optional[int]:
    """
    Load intent alignment result for a trace file.

    Args:
        file_path: Path to the trace file
        agent_name: Name of agent (banking, slack, travel, workspace)
        is_attack: True if attack trace, False if utility trace
        model_name: Model used for intent alignment (default: gpt-5-nano)
        intent_root: Root directory for intent alignment results

    Returns:
        Intent alignment result (0=aligned/utility, 1=misaligned/attack) or None if not found
    """
    # Map the trace file path to intent alignment result file
    # The drift evaluation creates filenames by replacing "/" with "_" in the full path
    # Input: /home/rohseque/agent_sentry/AD_run1/gpt-4o-2024-05-13/banking/user_task_0/none/none.json
    # Output: agentdojo_intent_alignment_results/AD_run1/evaluated_traces/gpt-5-mini/_home_rohseque_agent_sentry_AD_run1_gpt-4o-2024-05-13_banking_user_task_0_none_none_gpt-5-mini.json

    path_str = str(file_path)

    # Extract the AD_run dataset name from the path
    ad_run_dataset = None
    parts = Path(path_str).parts
    for part in parts:
        if part.startswith("AD_run"):
            ad_run_dataset = part
            break

    if not ad_run_dataset:
        return None

    # Create flattened filename by replacing slashes with underscores
    flat_name = path_str.replace("/", "_")

    # Add model suffix (match drift evaluation naming: replace dots and colons but keep hyphens)
    model_suffix = model_name.replace(".", "").replace(":", "_")
    name_part = flat_name.replace(".json", "")
    intent_filename = f"{name_part}_{model_suffix}.json"

    # Use custom intent_root if provided, otherwise use default
    alignment_root = Path(intent_root) if intent_root else Path(INTENT_ALIGNMENT_ROOT)

    # Construct full path
    # If intent_root is provided, it should already point to the AD_run directory
    # Otherwise, we need to append the ad_run_dataset
    if intent_root:
        intent_path = alignment_root / "evaluated_traces" / model_name / intent_filename
    else:
        intent_path = (
            alignment_root
            / ad_run_dataset
            / "evaluated_traces"
            / model_name
            / intent_filename
        )

    if not intent_path.exists():
        # Try without leading underscore (in case the path didn't start with /)
        if intent_filename.startswith("_"):
            alt_filename = intent_filename[1:]
            alt_path = (
                alignment_root
                / ad_run_dataset
                / "evaluated_traces"
                / model_name
                / alt_filename
            )
            if alt_path.exists():
                intent_path = alt_path
            else:
                return None
        else:
            return None

    try:
        with open(intent_path) as f:
            data = json.load(f)

        # Intent alignment returns a list with one item: [{"val": 0 or 1, ...}]
        if isinstance(data, list) and len(data) > 0:
            val = data[0].get("val")
            if val is not None:
                return int(val)
    except Exception as e:
        print(f"    Warning: Error loading intent alignment from {intent_path}: {e}")

    return None


def load_policy(policy_path, mode="heuristic"):
    """Load a trained policy from JSON file."""
    policy = ActionBlockingPolicy.load(policy_path)

    # Load threshold from metrics.json in the same directory
    metrics_path = policy_path.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
            policy.threshold = metrics.get("threshold", 0.0)
    else:
        policy.threshold = 0.0

    return policy


def load_trace(trace_path):
    """Load a trace file."""
    with open(trace_path) as f:
        return json.load(f)


def is_utility_trace(trace_path):
    """Check if trace is a valid utility trace (none.json with utility=True)."""
    if not trace_path.name == "none.json":
        return False

    try:
        trace = load_trace(trace_path)
        return trace.get("utility", False) is True
    except Exception as e:
        print(f"Warning: Error loading {trace_path}: {e}")
        return False


def is_attack_trace(trace_path):
    """Check if trace is a valid attack trace (under important_instructions with security=True)."""
    # Must be under important_instructions directory
    if "important_instructions" not in str(trace_path):
        return False

    # Must be a json file but not none.json
    if not trace_path.suffix == ".json" or trace_path.name == "none.json":
        return False

    try:
        trace = load_trace(trace_path)
        return trace.get("security", False) is True
    except Exception as e:
        print(f"Warning: Error loading {trace_path}: {e}")
        return False


def find_agentdojo_traces(data_dir):
    """
    Find all valid utility and attack traces in AgentDojo directory.

    Returns:
        dict: {agent_name: {'utility': [paths], 'attack': [paths]}}
    """
    data_path = Path(data_dir)
    traces = defaultdict(lambda: {"utility": [], "attack": []})

    # Expected agents
    agents = ["banking", "slack", "travel", "workspace"]

    for agent in agents:
        # Find all user_task_* directories for this agent
        # Try direct path first, then recursive
        agent_dir = data_path / agent
        if agent_dir.exists():
            agent_paths = list(agent_dir.glob("user_task_*"))
        else:
            agent_paths = list(data_path.rglob(f"*/{agent}/user_task_*"))

        for user_task_dir in agent_paths:
            if not user_task_dir.is_dir():
                continue

            # Check for utility trace (none/none.json or none.json)
            none_json = user_task_dir / "none" / "none.json"
            if not none_json.exists():
                none_json = user_task_dir / "none.json"
            if none_json.exists() and is_utility_trace(none_json):
                traces[agent]["utility"].append(none_json)

            # Check for attack traces (important_instructions/*.json)
            important_instructions = user_task_dir / "important_instructions"
            if important_instructions.exists() and important_instructions.is_dir():
                for attack_file in important_instructions.glob("*.json"):
                    if is_attack_trace(attack_file):
                        traces[agent]["attack"].append(attack_file)

    return traces


def evaluate_trace(trace, policy, extractor, agent_name, trace_path):
    """
    Evaluate a single trace against a policy.

    Returns:
        bool: True if blocked, False if allowed
    """
    try:
        # Extract trace using extractor to get DFG nodes
        trace_data = extractor.extract_from_file(str(trace_path), agent_name=agent_name)

        if not trace_data or not trace_data.tool_calls:
            return False  # No actions to evaluate, allow

        # Build tool sequence
        tool_seq = ["user_prompt"]
        for tc in trace_data.tool_calls:
            tool_seq.append(tc.name)

        # Track context for CFG
        actions_seen_so_far = []
        ret_tools_seen_so_far = set()

        max_score = float("-inf")

        # Score each tool in sequence
        for i, tool_name in enumerate(tool_seq):
            if tool_name == "user_prompt":
                continue

            # Update retrieval tools seen
            if tool_name in policy.retrieval_tools:
                ret_tools_seen_so_far.add(tool_name)
                continue

            # If it's an action tool, score it
            if tool_name in policy.action_tools:
                # Find the corresponding tool call object
                tc_index = i - 1
                if tc_index < 0 or tc_index >= len(trace_data.tool_calls):
                    continue

                tc = trace_data.tool_calls[tc_index]

                # Get DFG nodes for this tool call
                action_nodes = [
                    n for n in trace_data.dfg_nodes if n.tool_call_id == tc.tool_call_id
                ]

                # Score the action
                result = policy.score_action(
                    action_name=tool_name,
                    actions_seen_so_far=actions_seen_so_far,
                    ret_tools_seen_so_far=ret_tools_seen_so_far,
                    dfg_nodes=action_nodes,
                    detailed=False,
                )

                score = (
                    result
                    if isinstance(result, (int, float))
                    else result.get("score", 0.0)
                )
                max_score = max(max_score, score)

                # Update actions seen strings
                actions_seen_so_far.append(tool_name)

        # Check if trace should be blocked
        if max_score == float("-inf"):
            max_score = 0.0

        threshold = policy.threshold if hasattr(policy, "threshold") else 0.0
        return max_score > threshold

    except Exception as e:
        print(f"    Warning: Error evaluating trace {trace_path.name}: {e}")
        import traceback

        traceback.print_exc()
        return False  # Allow on error


def evaluate_agent(
    agent_name,
    policy,
    utility_traces,
    attack_traces,
    mode="heuristic",
    intent_model="gpt-5-nano",
    intent_root=None,
):
    """Evaluate a single agent's policy on its traces."""

    results = {
        "agent": agent_name,
        "utility_total": len(utility_traces),
        "utility_allowed": 0,
        "utility_blocked": 0,
        "attack_total": len(attack_traces),
        "attack_allowed": 0,
        "attack_blocked": 0,
        # Combined system (policy OR intent=1)
        "combined_utility_allowed": 0,
        "combined_utility_blocked": 0,
        "combined_attack_allowed": 0,
        "combined_attack_blocked": 0,
        "intent_available": 0,
    }

    # Create extractor for this agent
    extractor = TraceExtractor(mode=mode, debug=False)

    # Evaluate utility traces (should be allowed)
    for trace_path in utility_traces:
        try:
            trace = load_trace(trace_path)
            policy_blocked = evaluate_trace(
                trace, policy, extractor, agent_name, trace_path
            )

            # Policy-only results
            if policy_blocked:
                results["utility_blocked"] += 1
            else:
                results["utility_allowed"] += 1

            # Combined system: block if policy blocks OR intent=1 (attack)
            intent_result = load_intent_alignment_result(
                str(trace_path),
                agent_name,
                is_attack=False,
                model_name=intent_model,
                intent_root=intent_root,
            )
            if intent_result is not None:
                results["intent_available"] += 1
                combined_blocked = policy_blocked or (intent_result == 1)
            else:
                # If no intent available, fall back to policy only
                combined_blocked = policy_blocked

            if combined_blocked:
                results["combined_utility_blocked"] += 1
            else:
                results["combined_utility_allowed"] += 1

        except Exception as e:
            print(f"  Error evaluating utility trace {trace_path.name}: {e}")

    # Evaluate attack traces (should be blocked)
    for trace_path in attack_traces:
        try:
            trace = load_trace(trace_path)
            policy_blocked = evaluate_trace(
                trace, policy, extractor, agent_name, trace_path
            )

            # Policy-only results
            if policy_blocked:
                results["attack_blocked"] += 1
            else:
                results["attack_allowed"] += 1

            # Combined system: block if policy blocks OR intent=1 (attack)
            intent_result = load_intent_alignment_result(
                str(trace_path),
                agent_name,
                is_attack=True,
                model_name=intent_model,
                intent_root=intent_root,
            )
            if intent_result is not None:
                results["intent_available"] += 1
                combined_blocked = policy_blocked or (intent_result == 1)
            else:
                # If no intent available, fall back to policy only
                combined_blocked = policy_blocked

            if combined_blocked:
                results["combined_attack_blocked"] += 1
            else:
                results["combined_attack_allowed"] += 1

        except Exception as e:
            print(f"  Error evaluating attack trace {trace_path.name}: {e}")

    # Calculate rates
    results["utility_success_rate"] = (
        results["utility_allowed"] / results["utility_total"] * 100
        if results["utility_total"] > 0
        else 0.0
    )
    results["attack_block_rate"] = (
        results["attack_blocked"] / results["attack_total"] * 100
        if results["attack_total"] > 0
        else 0.0
    )

    # Combined system rates
    results["combined_utility_success_rate"] = (
        results["combined_utility_allowed"] / results["utility_total"] * 100
        if results["utility_total"] > 0
        else 0.0
    )
    results["combined_attack_block_rate"] = (
        results["combined_attack_blocked"] / results["attack_total"] * 100
        if results["attack_total"] > 0
        else 0.0
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate policies on AgentDojo dataset"
    )
    parser.add_argument(
        "--policy_dir",
        type=str,
        required=True,
        help="Directory containing trained policies (e.g., experiments/slack_optimized_r4_d2)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="AgentDojo results directory (e.g., /home/rohseque/agent_sentry/AD_run1)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "semantic"],
        help="Mode for DFG extraction: 'heuristic' or 'semantic' (default: heuristic)",
    )
    parser.add_argument(
        "--output", type=str, help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--intent_model",
        type=str,
        default="gpt-5-nano",
        help="Model used for intent alignment (default: gpt-5-nano)",
    )
    parser.add_argument(
        "--intent_root",
        type=str,
        default=None,
        help="Root directory for intent alignment results (default: intent_alignment/evaluated_traces)",
    )

    args = parser.parse_args()

    policy_dir = Path(args.policy_dir)
    data_dir = Path(args.data_dir)

    if not policy_dir.exists():
        print(f"Error: Policy directory not found: {policy_dir}")
        return 1

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    print("=" * 80)
    print("AGENTDOJO POLICY EVALUATION")
    print("=" * 80)
    print(f"\nPolicy directory: {policy_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Mode: {args.mode}")
    print()

    # Find all traces
    print("Finding AgentDojo traces...")
    traces = find_agentdojo_traces(data_dir)

    if not traces:
        print("Error: No valid traces found!")
        return 1

    print(f"\nFound traces:")
    for agent, agent_traces in sorted(traces.items()):
        print(
            f"  {agent}: {len(agent_traces['utility'])} utility, {len(agent_traces['attack'])} attack"
        )

    # Evaluate each agent
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    all_results = []

    for agent in sorted(traces.keys()):
        print(f"\n{'='*80}")
        print(f"Agent: {agent}")
        print(f"{'='*80}")

        # Load policy
        # Try multiple possible policy paths
        policy_path = None
        possible_paths = [
            policy_dir / agent / "train_frac_100" / "policy.json",
            policy_dir / agent / "policy.json",
            policy_dir / f"{agent}_policy.json",
        ]

        for path in possible_paths:
            if path.exists():
                policy_path = path
                break

        if not policy_path:
            print(f"  Error: Could not find policy for {agent}")
            print(f"  Tried: {[str(p) for p in possible_paths]}")
            continue

        print(f"Loading policy: {policy_path}")
        policy = load_policy(policy_path, mode=args.mode)
        print(
            f"Policy threshold: {policy.threshold if hasattr(policy, 'threshold') else 'N/A'}"
        )

        # Evaluate
        results = evaluate_agent(
            agent,
            policy,
            traces[agent]["utility"],
            traces[agent]["attack"],
            mode=args.mode,
            intent_model=args.intent_model,
            intent_root=args.intent_root,
        )

        all_results.append(results)

        # Print results
        print(f"\n{'='*40}")
        print(f"POLICY ONLY")
        print(f"{'='*40}")
        print(f"\nUtility Traces:")
        print(f"  Total: {results['utility_total']}")
        print(f"  Allowed: {results['utility_allowed']} (should allow)")
        print(f"  Blocked: {results['utility_blocked']} (false positives)")
        print(f"  Success Rate: {results['utility_success_rate']:.2f}%")

        print(f"\nAttack Traces:")
        print(f"  Total: {results['attack_total']}")
        print(f"  Blocked: {results['attack_blocked']} (should block)")
        print(f"  Allowed: {results['attack_allowed']} (false negatives)")
        print(f"  Block Rate: {results['attack_block_rate']:.2f}%")

        print(f"\n{'='*40}")
        print(f"COMBINED SYSTEM (Policy OR Intent=1)")
        print(f"{'='*40}")
        print(f"\nUtility Traces:")
        print(f"  Allowed: {results['combined_utility_allowed']}")
        print(f"  Blocked: {results['combined_utility_blocked']}")
        print(f"  Success Rate: {results['combined_utility_success_rate']:.2f}%")

        print(f"\nAttack Traces:")
        print(f"  Blocked: {results['combined_attack_blocked']}")
        print(f"  Allowed: {results['combined_attack_allowed']}")
        print(f"  Block Rate: {results['combined_attack_block_rate']:.2f}%")

        if results["intent_available"] == 0:
            print(
                f"\n  Note: No intent alignment data available - combined system same as policy"
            )

    # Summary
    print("\n" + "=" * 80)
    print("POLICY ONLY SUMMARY")
    print("=" * 80)
    print()
    print(
        f"{'Agent':<12} {'Utility %':<12} {'Attack Block %':<15} {'Utility Total':<15} {'Attack Total':<15}"
    )
    print("-" * 80)

    for result in all_results:
        print(
            f"{result['agent']:<12} {result['utility_success_rate']:<12.2f} "
            f"{result['attack_block_rate']:<15.2f} {result['utility_total']:<15} "
            f"{result['attack_total']:<15}"
        )

    # Calculate averages
    if all_results:
        avg_utility = sum(r["utility_success_rate"] for r in all_results) / len(
            all_results
        )
        avg_attack = sum(r["attack_block_rate"] for r in all_results) / len(all_results)

        print("-" * 80)
        print(f"{'AVERAGE':<12} {avg_utility:<12.2f} {avg_attack:<15.2f}")

    # Combined system summary
    print("\n" + "=" * 80)
    print("COMBINED SYSTEM (Policy OR Intent=1) SUMMARY")
    print("=" * 80)
    print()
    print(
        f"{'Agent':<12} {'Utility %':<12} {'Attack Block %':<15} {'Utility Total':<15} {'Attack Total':<15}"
    )
    print("-" * 80)

    for result in all_results:
        print(
            f"{result['agent']:<12} {result['combined_utility_success_rate']:<12.2f} "
            f"{result['combined_attack_block_rate']:<15.2f} {result['utility_total']:<15} "
            f"{result['attack_total']:<15}"
        )

    # Calculate combined averages
    if all_results:
        avg_combined_utility = sum(
            r["combined_utility_success_rate"] for r in all_results
        ) / len(all_results)
        avg_combined_attack = sum(
            r["combined_attack_block_rate"] for r in all_results
        ) / len(all_results)

        print("-" * 80)
        print(
            f"{'AVERAGE':<12} {avg_combined_utility:<12.2f} {avg_combined_attack:<15.2f}"
        )

        total_intent = sum(r["intent_available"] for r in all_results)
        if total_intent == 0:
            print(
                "\nNote: No intent alignment data available - combined system same as policy"
            )

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
