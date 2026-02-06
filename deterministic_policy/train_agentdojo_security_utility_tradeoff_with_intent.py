#!/usr/bin/env python3
"""
Security-Utility Tradeoff Experiment for AgentDojo Dataset

Trains ActionBlockingPolicy models on varying fractions of utility data (10%-100%)
and evaluates them on full utility and attack datasets to measure security-utility tradeoffs.

This version is specifically adapted for AgentDojo dataset format.

Usage:
    python train_agentdojo_security_utility_tradeoff_with_intent.py \\
        --dataset_root /home/rohseque/agent_sentry/AD_run1/gpt-4o-2024-05-13 \\
        --intent_root /home/rohseque/agent_sentry/agentdojo_intent_alignment_results/AD_run1 \\
        --output_dir experiments/agentdojo_security_utility_tradeoff \\
        --mode heuristic \\
        --workers 100 \\
        --seed 42
"""

import os
import json
import csv
import random
import argparse
import time
import math
import numpy as np
from glob import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

# Import existing codebase components
from policy import ActionBlockingPolicy
from extractor import TraceExtractor
from structures import TraceData

# ============================================================================
# Configuration
# ============================================================================

AGENTS = ["banking", "travel", "slack", "workspace"]
TRAINING_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [42, 43, 44]  # Multiple seeds for reproducibility

DEFAULT_HYPERPARAMETERS = {
    "beta": 1.0,
    "unseen_penalty": 10.0,
    "cfg_state_penalty_common": 0.0,
    "cfg_state_penalty_rare": 1.5,
    "cfg_state_penalty_unseen": 5.0,
    "cfg_action_penalty_common": 0.0,
    "cfg_action_penalty_rare": 2.0,
    "cfg_action_penalty_unseen": 5.0,
}


# ============================================================================
# AgentDojo Trace Identification
# ============================================================================


def is_utility_trace(trace_path: Path) -> bool:
    """
    Check if trace is a valid utility trace (none.json with utility=True).

    Args:
        trace_path: Path to trace file

    Returns:
        True if valid utility trace, False otherwise
    """
    if not trace_path.name == "none.json":
        return False

    try:
        with open(trace_path) as f:
            trace = json.load(f)
        return trace.get("utility", False) is True
    except Exception as e:
        print(f"Warning: Error loading {trace_path}: {e}")
        return False


def is_attack_trace(trace_path: Path) -> bool:
    """
    Check if trace is a valid attack trace (under important_instructions with security=True).

    Args:
        trace_path: Path to trace file

    Returns:
        True if valid attack trace, False otherwise
    """
    # Must be under important_instructions directory
    if "important_instructions" not in str(trace_path):
        return False

    # Must be a json file but not none.json
    if not trace_path.suffix == ".json" or trace_path.name == "none.json":
        return False

    try:
        with open(trace_path) as f:
            trace = json.load(f)
        return trace.get("security", False) is True
    except Exception as e:
        print(f"Warning: Error loading {trace_path}: {e}")
        return False


def find_agentdojo_traces(data_dir: Path, agent_name: str) -> Dict[str, List[Path]]:
    """
    Find all valid utility and attack traces for an agent in AgentDojo directory.

    Args:
        data_dir: Root directory of AgentDojo dataset
        agent_name: Name of agent (banking, slack, travel, workspace)

    Returns:
        Dict with 'utility' and 'attack' lists of trace paths
    """
    traces = {"utility": [], "attack": []}

    # Find all user_task_* directories for this agent
    agent_paths = list(data_dir.rglob(f"{agent_name}/user_task_*"))

    for user_task_dir in agent_paths:
        if not user_task_dir.is_dir():
            continue

        # Check for utility trace (none/none.json)
        none_json = user_task_dir / "none" / "none.json"
        if none_json.exists() and is_utility_trace(none_json):
            traces["utility"].append(none_json)

        # Check for attack traces (important_instructions/*.json)
        important_instructions = user_task_dir / "important_instructions"
        if important_instructions.exists() and important_instructions.is_dir():
            for attack_file in important_instructions.glob("*.json"):
                if is_attack_trace(attack_file):
                    traces["attack"].append(attack_file)

    return traces


# ============================================================================
# Caching System
# ============================================================================


def get_cache_path(
    dataset_root: str, file_path: str, cache_root: str, mode: str
) -> str:
    """
    Get cache path for extracted trace, mirroring dataset structure.

    Args:
        dataset_root: Root directory of dataset
        file_path: Path to original trace file
        cache_root: Root directory for cache
        mode: Extraction mode (heuristic/semantic/hybrid)

    Returns:
        Path to cache file
    """
    # For AgentDojo, include the full path structure in cache
    rel_path = os.path.relpath(file_path, dataset_root)
    return os.path.join(cache_root, mode, "agentdojo", rel_path)


def extract_trace_cached(
    file_path: str,
    dataset_root: str,
    agent_name: str,
    mode: str,
    cache_root: str = "cache",
) -> Optional[TraceData]:
    """
    Extract trace with caching support.

    Args:
        file_path: Path to trace JSON file
        dataset_root: Root directory of dataset
        agent_name: Name of agent (banking, slack, etc.)
        mode: Extraction mode (heuristic, semantic, hybrid)
        cache_root: Root directory for cache

    Returns:
        TraceData object or None if extraction fails
    """
    cache_path = get_cache_path(dataset_root, file_path, cache_root, mode)

    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return TraceData.from_dict(json.load(f))
        except Exception as e:
            print(
                f"  Warning: Failed to load cache for {os.path.basename(file_path)}: {e}"
            )

    # Extract if not cached
    try:
        extractor = TraceExtractor(mode=mode, debug=False)
        trace = extractor.extract_from_file(file_path, agent_name=agent_name)

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)

        return trace
    except Exception as e:
        print(f"  Warning: Failed to extract {os.path.basename(file_path)}: {e}")
        return None


# ============================================================================
# Intent Alignment Integration
# ============================================================================


def load_intent_alignment_result(
    file_path: str, agent_name: str, intent_root: str, model_name: str = "gpt-5-mini"
) -> Optional[int]:
    """
    Load intent alignment result for a trace file from AgentDojo intent alignment results.

    Args:
        file_path: Path to the original trace file
        agent_name: Name of agent (banking, slack, travel, workspace)
        intent_root: Root directory for intent alignment results
        model_name: Model used for intent alignment

    Returns:
        Intent alignment result (0=aligned/utility, 1=misaligned/attack) or None if not found
    """
    # Create flattened filename by replacing slashes with underscores
    flat_name = file_path.replace("/", "_")

    # Add model suffix
    model_suffix = model_name.replace(".", "").replace(":", "_")
    name_part = flat_name.replace(".json", "")
    intent_filename = f"{name_part}_{model_suffix}.json"

    # Construct path: intent_root / evaluated_traces / model_name / filename
    intent_path = Path(intent_root) / "evaluated_traces" / model_name / intent_filename

    if not intent_path.exists():
        # Try without leading underscore
        if intent_filename.startswith("_"):
            alt_filename = intent_filename[1:]
            alt_path = (
                Path(intent_root) / "evaluated_traces" / model_name / alt_filename
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
        print(f"  Warning: Error loading intent alignment from {intent_path}: {e}")

    return None


def combine_results(policy_blocked: bool, intent_misaligned: bool, logic: str) -> bool:
    """
    Combine policy and intent alignment results using specified logic.

    Args:
        policy_blocked: True if policy blocked the action
        intent_misaligned: True if intent alignment detected misalignment (val=1)
        logic: 'OR', 'AND', or 'XOR'

    Returns:
        Combined blocking decision
    """
    if logic == "OR":
        return policy_blocked or intent_misaligned
    elif logic == "AND":
        return policy_blocked and intent_misaligned
    elif logic == "XOR":
        return policy_blocked != intent_misaligned
    else:
        raise ValueError(f"Unknown logic: {logic}")


# ============================================================================
# Data Loading
# ============================================================================


def load_utility_traces(agent_name: str, dataset_root: str) -> List[str]:
    """
    Load all utility trace file paths for an agent from AgentDojo dataset.

    Args:
        agent_name: Name of agent
        dataset_root: Root directory of dataset

    Returns:
        List of file paths to utility traces
    """
    traces = find_agentdojo_traces(Path(dataset_root), agent_name)
    return [str(p) for p in traces["utility"]]


def load_attack_traces(agent_name: str, dataset_root: str) -> List[str]:
    """
    Load all attack trace file paths for an agent from AgentDojo dataset.

    Args:
        agent_name: Name of agent
        dataset_root: Root directory of dataset

    Returns:
        List of file paths to attack traces
    """
    traces = find_agentdojo_traces(Path(dataset_root), agent_name)
    return [str(p) for p in traces["attack"]]


# ============================================================================
# Training Fraction Sampling
# ============================================================================


def sample_training_fraction(
    all_files: List[str], fraction: float, seed: int
) -> List[str]:
    """
    Sample a fraction of files for training.

    Args:
        all_files: List of all available file paths
        fraction: Fraction to sample (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of sampled file paths
    """
    rng = random.Random(seed)
    sample_size = max(1, int(len(all_files) * fraction))
    return rng.sample(all_files, sample_size)


# ============================================================================
# Parallel Extraction
# ============================================================================


def extract_traces_parallel(
    file_paths: List[str],
    dataset_root: str,
    agent_name: str,
    mode: str,
    cache_root: str,
    workers: int,
) -> List[TraceData]:
    """
    Extract multiple traces in parallel with caching.

    Args:
        file_paths: List of trace file paths
        dataset_root: Root directory of dataset
        agent_name: Name of agent
        mode: Extraction mode
        cache_root: Root directory for cache
        workers: Number of parallel workers

    Returns:
        List of successfully extracted TraceData objects
    """
    traces = []

    def extract_one(fpath):
        return extract_trace_cached(fpath, dataset_root, agent_name, mode, cache_root)

    # Run sequentially for now (parallel can be enabled later)
    for fpath in file_paths:
        trace = extract_one(fpath)
        if trace is not None and trace.tool_calls:
            traces.append(trace)

    return traces


# ============================================================================
# Cache Pre-building
# ============================================================================


def build_input_cache(
    agents: List[str],
    dataset_root: str,
    mode: str,
    cache_root: str,
    workers: int,
):
    """
    Pre-build cache for ALL utility and attack traces.

    Args:
        agents: List of agents to process
        dataset_root: Root directory of dataset
        mode: Extraction mode
        cache_root: Root directory for cache
        workers: Number of parallel workers
    """
    print(f"\nPre-building cache for {len(agents)} agents (Mode: {mode})...")

    def needs_extraction(fpath):
        cache_path = get_cache_path(dataset_root, fpath, cache_root, mode)
        return not os.path.exists(cache_path)

    total_extracted = 0
    for agent in agents:
        util_files = load_utility_traces(agent, dataset_root)
        att_files = load_attack_traces(agent, dataset_root)

        agent_files = []
        agent_files.extend([f for f in util_files if needs_extraction(f)])
        agent_files.extend([f for f in att_files if needs_extraction(f)])

        if not agent_files:
            continue

        print(f"  {agent}: Extracting {len(agent_files)} traces...")

        extract_traces_parallel(
            agent_files, dataset_root, agent, mode, cache_root, workers
        )
        total_extracted += len(agent_files)

    if total_extracted == 0:
        print("Cache is up to date!")
    else:
        print(f"Cache build complete! Processed {total_extracted} traces.\n")


# ============================================================================
# Policy Scoring and Evaluation
# ============================================================================


def score_trace_with_policy(
    policy: ActionBlockingPolicy,
    trace: TraceData,
    threshold: float,
) -> Tuple[float, bool, Dict]:
    """
    Score a single trace and determine if it would be blocked.

    Args:
        policy: Trained policy
        trace: Trace to score
        threshold: Blocking threshold

    Returns:
        (max_score, is_blocked, details_dict) tuple
    """

    max_score = float("-inf")
    action_details = []

    # Build tool sequence
    tool_seq = ["user_prompt"]
    for tc in trace.tool_calls:
        tool_seq.append(tc.name)

    # Track context for CFG
    actions_seen_so_far = []
    ret_tools_seen_so_far = set()

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
            # We need to find the K-th occurrence of this tool name if there are duplicates
            # A simple approach is to iterate through trace.tool_calls and match by index
            # tool_seq is ["user_prompt", tc1, tc2, ...]. So index i corresponds to trace.tool_calls[i-1]

            tc_index = i - 1
            if tc_index < 0 or tc_index >= len(trace.tool_calls):
                continue

            tc = trace.tool_calls[tc_index]

            # Get DFG nodes for this tool call
            # The policy expects all DFG nodes, it filters internally?
            # Looking at policy.py: score_action(..., dfg_nodes, ...) -> _score_dfg(action_name, dfg_nodes) -> iterates dfg_nodes
            # It seems to expect the FULL list of DFG nodes for the trace, or just for the action?
            # _score_dfg iterates all nodes and checks `if node.tool_name == action_name`.
            # Wait, if there are multiple calls to same action, `node.tool_name` is ambiguous?
            # policy.py doesn't seem to filter by tool_call_id in `_score_dfg`.
            # Let's check policy.py again.
            # `for node in dfg_nodes: if node.tool_name == action_name: ...`
            # Yes, it matches by NAME. This might be a limitation in policy.py or intended design (context-free DFG scoring per action type?).
            # However, here we have `tc.tool_call_id`.
            # If I pass ONLY the nodes for this specific tool call, it's safer.
            action_nodes = [
                n for n in trace.dfg_nodes if n.tool_call_id == tc.tool_call_id
            ]

            # Score the action
            # policy.score_action(action_name, actions_seen, ret_tools_seen, dfg_nodes, detailed=True)
            result = policy.score_action(
                action_name=tool_name,
                actions_seen_so_far=actions_seen_so_far,
                ret_tools_seen_so_far=ret_tools_seen_so_far,
                dfg_nodes=action_nodes,  # Pass specific nodes to avoid ambiguity
                detailed=True,
            )

            score = result.get("score", 0.0)
            max_score = max(max_score, score)

            action_details.append(
                {
                    "action_name": tool_name,
                    "action_score": float(score),
                    "details": result,
                }
            )

            # Update actions seen strings
            actions_seen_so_far.append(tool_name)

    if max_score == float("-inf"):
        max_score = 0.0

    is_blocked = max_score > threshold

    details = {
        "max_score": float(max_score),
        "is_blocked": is_blocked,
        "threshold": float(threshold),
        "num_actions": len(action_details),
        "actions": action_details,
    }

    return max_score, is_blocked, details


def evaluate_utility_rate(
    policy: ActionBlockingPolicy,
    traces: List[TraceData],
    threshold: float,
) -> float:
    """
    Evaluate utility success rate on traces.

    Args:
        policy: Trained policy
        traces: List of utility traces
        threshold: Blocking threshold
        untrusted_retrieval_tools: Set of untrusted retrieval tools

    Returns:
        Utility success rate (0-100)
    """
    if not traces:
        return 0.0

    allowed = 0
    for trace in traces:
        _, is_blocked, _ = score_trace_with_policy(policy, trace, threshold)
        if not is_blocked:
            allowed += 1

    return (allowed / len(traces)) * 100.0


# ============================================================================
# Policy Training
# ============================================================================


def train_policy_on_fraction(
    agent_name: str,
    training_files: List[str],
    attack_files: List[str],
    fraction: float,
    dataset_root: str,
    output_dir: str,
    mode: str,
    workers: int,
    cache_root: str,
    seed: int,
    target_utility_rate: float = 90.0,
) -> Tuple[ActionBlockingPolicy, Dict]:
    """
    Train a policy on a fraction of utility data with hyperparameter tuning.

    Args:
        agent_name: Name of agent
        training_files: List of training file paths
        attack_files: List of attack file paths (required for deterministic policy)
        fraction: Training fraction used
        dataset_root: Root directory of dataset
        output_dir: Output directory for this experiment
        mode: Extraction mode
        workers: Number of parallel workers
        cache_root: Cache root directory
        seed: Random seed
        target_utility_rate: Target utility rate for threshold search

    Returns:
        (trained_policy, training_metadata)
    """
    print(f"\n{'='*80}")
    print(
        f"Training {agent_name} on {fraction*100:.0f}% of utility data ({len(training_files)} traces)"
    )
    print(f"and {len(attack_files)} attack traces")
    print(f"{'='*80}")

    # Extract training traces
    print(f"Extracting {len(training_files)} training traces...")
    training_traces = extract_traces_parallel(
        training_files, dataset_root, agent_name, mode, cache_root, workers
    )

    print(f"Extracting {len(attack_files)} attack traces...")
    attack_traces = extract_traces_parallel(
        attack_files, dataset_root, agent_name, mode, cache_root, workers
    )

    if not training_traces:
        print(f"  ERROR: No valid training traces extracted!")
        return None, {}

    if not attack_traces:
        print(f"  ERROR: No valid attack traces extracted!")
        return None, {}

    print(
        f"  Successfully extracted {len(training_traces)} utility and {len(attack_traces)} attack traces"
    )

    # Load tool classifications for this agent
    tool_class_path = os.path.join("utils", agent_name, "tool_classification.json")
    action_tools = set()
    retrieval_tools = set()

    if os.path.exists(tool_class_path):
        with open(tool_class_path) as f:
            tool_class = json.load(f)
            action_tools = set(tool_class.get("action_tools", []))
            retrieval_tools = set(tool_class.get("retrieval_tools", []))
            print(
                f"  Loaded tool classifications: {len(action_tools)} action, {len(retrieval_tools)} retrieval"
            )
    else:
        print(f"  WARNING: No tool classification found at {tool_class_path}")

    # Train policy
    print(f"Training policy...")
    policy = ActionBlockingPolicy()

    # Set tool classifications
    policy.action_tools = action_tools
    policy.retrieval_tools = retrieval_tools

    # Train with both utility and attack traces
    policy.train(utility_traces=training_traces, attack_traces=attack_traces)

    # Find threshold that achieves target utility rate
    print(f"Finding threshold for {target_utility_rate}% utility rate...")

    # Test multiple thresholds
    # Deterministic policy scores are integers/floats like -10, -5, 0, 2, 3, 5, 10
    # Range is roughly -15 to +15.
    thresholds = [i for i in range(-20, 21)]
    best_threshold = 0.0
    best_utility = 0.0
    best_diff = float("inf")

    for thresh in thresholds:
        util_rate = evaluate_utility_rate(policy, training_traces, thresh)
        diff = abs(util_rate - target_utility_rate)

        # Find threshold closest to target
        if diff < best_diff:
            best_diff = diff
            best_threshold = thresh
            best_utility = util_rate

    print(
        f"  Selected threshold: {best_threshold:.2f} (achieves {best_utility:.2f}% utility)"
    )

    metadata = {
        "agent": agent_name,
        "training_fraction": fraction,
        "training_samples": len(training_traces),
        "num_attack_traces": len(attack_traces),
        "threshold": best_threshold,
        "training_utility_rate": best_utility,
        "target_utility_rate": target_utility_rate,
    }

    return policy, metadata


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_combined_system(
    agent_name: str,
    policy: ActionBlockingPolicy,
    threshold: float,
    utility_files: List[str],
    attack_files: List[str],
    dataset_root: str,
    intent_root: str,
    mode: str,
    cache_root: str,
    workers: int,
    logic: str = "OR",
) -> Dict:
    """
    Evaluate combined system (policy + intent alignment) on full dataset.

    Args:
        agent_name: Name of agent
        policy: Trained policy
        threshold: Blocking threshold
        utility_files: List of utility file paths
        attack_files: List of attack file paths
        dataset_root: Root directory of dataset
        intent_root: Root directory for intent alignment results
        mode: Extraction mode
        cache_root: Cache root directory
        workers: Number of parallel workers
        logic: Combination logic ('OR', 'AND', 'XOR')

    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating {agent_name} on full dataset...")

    # Load tool classifications for untrusted retrieval tools

    # Extract all traces
    print(f"  Extracting {len(utility_files)} utility traces...")
    utility_traces = extract_traces_parallel(
        utility_files, dataset_root, agent_name, mode, cache_root, workers
    )

    print(f"  Extracting {len(attack_files)} attack traces...")
    attack_traces = extract_traces_parallel(
        attack_files, dataset_root, agent_name, mode, cache_root, workers
    )

    results = {
        "agent": agent_name,
        # Policy-only results
        "policy_utility_total": len(utility_traces),
        "policy_utility_allowed": 0,
        "policy_utility_blocked": 0,
        "policy_attack_total": len(attack_traces),
        "policy_attack_allowed": 0,
        "policy_attack_blocked": 0,
        # Combined system
        "combined_utility_total": len(utility_traces),
        "combined_utility_allowed": 0,
        "combined_utility_blocked": 0,
        "combined_utility_allowed_or": 0,
        "combined_utility_blocked_or": 0,
        "combined_utility_allowed_and": 0,
        "combined_utility_blocked_and": 0,
        "combined_attack_total": len(attack_traces),
        "combined_attack_allowed": 0,
        "combined_attack_blocked": 0,
        "combined_attack_allowed_or": 0,
        "combined_attack_blocked_or": 0,
        "combined_attack_allowed_and": 0,
        "combined_attack_blocked_and": 0,
        # Intent alignment availability
        "intent_available_utility": 0,
        "intent_available_attack": 0,
    }

    # Evaluate utility traces
    utility_scores = []
    for trace_file, trace in zip(utility_files, utility_traces):
        max_score, policy_blocked, _ = score_trace_with_policy(policy, trace, threshold)
        utility_scores.append(max_score)

        # Policy-only
        if policy_blocked:
            results["policy_utility_blocked"] += 1
        else:
            results["policy_utility_allowed"] += 1

        # Combined system
        intent_result = load_intent_alignment_result(
            trace_file, agent_name, intent_root
        )

        combined_blocked_or = policy_blocked
        combined_blocked_and = policy_blocked
        combined_blocked_legacy = policy_blocked

        if intent_result is not None:
            results["intent_available_utility"] += 1
            intent_misaligned = intent_result == 1

            # Policy rules:
            # 1. If score > 2.0 (high confidence attack), intent is ignored -> use policy decision (blocked)
            # 2. If score <= 2.0 (low confidence/ambiguous), use combined logic
            if max_score > 2.0:
                combined_blocked_or = policy_blocked
                combined_blocked_and = policy_blocked
                combined_blocked_legacy = policy_blocked
            else:
                combined_blocked_or = combine_results(
                    policy_blocked, intent_misaligned, "OR"
                )
                combined_blocked_and = combine_results(
                    policy_blocked, intent_misaligned, "AND"
                )
                combined_blocked_legacy = combine_results(
                    policy_blocked, intent_misaligned, logic
                )
        else:
            combined_blocked_or = policy_blocked
            combined_blocked_and = policy_blocked
            combined_blocked_legacy = policy_blocked

        # Update OR counters
        if combined_blocked_or:
            results["combined_utility_blocked_or"] += 1
        else:
            results["combined_utility_allowed_or"] += 1

        # Update AND counters
        if combined_blocked_and:
            results["combined_utility_blocked_and"] += 1
        else:
            results["combined_utility_allowed_and"] += 1

        # Update Legacy/Indicated Logic counters
        if combined_blocked_legacy:
            results["combined_utility_blocked"] += 1
        else:
            results["combined_utility_allowed"] += 1

    # Evaluate attack traces
    attack_scores = []
    for trace_file, trace in zip(attack_files, attack_traces):
        max_score, policy_blocked, _ = score_trace_with_policy(policy, trace, threshold)
        attack_scores.append(max_score)

        # Policy-only
        if policy_blocked:
            results["policy_attack_blocked"] += 1
        else:
            results["policy_attack_allowed"] += 1

        # Combined system
        intent_result = load_intent_alignment_result(
            trace_file, agent_name, intent_root
        )

        combined_blocked_or = policy_blocked
        combined_blocked_and = policy_blocked
        combined_blocked_legacy = policy_blocked

        if intent_result is not None:
            results["intent_available_attack"] += 1
            intent_misaligned = intent_result == 1

            if max_score > 2.0:
                combined_blocked_or = policy_blocked
                combined_blocked_and = policy_blocked
                combined_blocked_legacy = policy_blocked
            else:
                combined_blocked_or = combine_results(
                    policy_blocked, intent_misaligned, "OR"
                )
                combined_blocked_and = combine_results(
                    policy_blocked, intent_misaligned, "AND"
                )
                combined_blocked_legacy = combine_results(
                    policy_blocked, intent_misaligned, logic
                )
        else:
            combined_blocked_or = policy_blocked
            combined_blocked_and = policy_blocked
            combined_blocked_legacy = policy_blocked

        # Update OR counters
        if combined_blocked_or:
            results["combined_attack_blocked_or"] += 1
        else:
            results["combined_attack_allowed_or"] += 1

        # Update AND counters
        if combined_blocked_and:
            results["combined_attack_blocked_and"] += 1
        else:
            results["combined_attack_allowed_and"] += 1

        # Update Legacy/Indicated Logic counters
        if combined_blocked_legacy:
            results["combined_attack_blocked"] += 1
        else:
            results["combined_attack_allowed"] += 1

    # Calculate rates
    results["policy_utility_rate"] = (
        results["policy_utility_allowed"] / results["policy_utility_total"] * 100
        if results["policy_utility_total"] > 0
        else 0.0
    )
    results["policy_attack_block_rate"] = (
        results["policy_attack_blocked"] / results["policy_attack_total"] * 100
        if results["policy_attack_total"] > 0
        else 0.0
    )

    # Combined OR Rates
    results["combined_utility_rate_or"] = (
        results["combined_utility_allowed_or"] / results["combined_utility_total"] * 100
        if results["combined_utility_total"] > 0
        else 0.0
    )
    results["combined_attack_block_rate_or"] = (
        results["combined_attack_blocked_or"] / results["combined_attack_total"] * 100
        if results["combined_attack_total"] > 0
        else 0.0
    )

    # Combined AND Rates
    results["combined_utility_rate_and"] = (
        results["combined_utility_allowed_and"]
        / results["combined_utility_total"]
        * 100
        if results["combined_utility_total"] > 0
        else 0.0
    )
    results["combined_attack_block_rate_and"] = (
        results["combined_attack_blocked_and"] / results["combined_attack_total"] * 100
        if results["combined_attack_total"] > 0
        else 0.0
    )

    # Legacy Rates (based on `logic` arg)
    results["combined_utility_rate"] = (
        results["combined_utility_allowed"] / results["combined_utility_total"] * 100
        if results["combined_utility_total"] > 0
        else 0.0
    )
    results["combined_attack_block_rate"] = (
        results["combined_attack_blocked"] / results["combined_attack_total"] * 100
        if results["combined_attack_total"] > 0
        else 0.0
    )

    print(
        f"\n  Policy-only: Utility={results['policy_utility_rate']:.1f}%, Attack Block={results['policy_attack_block_rate']:.1f}%"
    )
    print(
        f"  Combined ({logic}): Utility={results['combined_utility_rate']:.1f}%, Attack Block={results['combined_attack_block_rate']:.1f}%"
    )
    print(
        f"  Intent availability: Utility={results['intent_available_utility']}/{results['policy_utility_total']}, Attack={results['intent_available_attack']}/{results['policy_attack_total']}"
    )

    return results, utility_scores, attack_scores


# ============================================================================
# Main Experiment
# ============================================================================


# ============================================================================
# Aggregation Across Seeds
# ============================================================================


def aggregate_results(
    results_list: List[Dict],  # List of result bundles (one per seed)
    agent: str,
    fraction: float,
    output_dir: str,
):
    """
    Aggregates results across seeds and saves summaries.
    
    Args:
        results_list: List of result dictionaries (one per seed)
        agent: Agent name
        fraction: Training fraction
        output_dir: Output directory
        
    Returns:
        Aggregated summary dictionary
    """
    if not results_list:
        return None
    
    summary_dir = os.path.join(
        output_dir, agent, f"train_frac_{int(fraction*100)}", "aggregated"
    )
    os.makedirs(summary_dir, exist_ok=True)

    agg = {}
    
    # Calculate means and stds for percentage metrics
    percentage_metrics = [
        "policy_utility_rate",
        "policy_attack_block_rate",
        "combined_utility_rate",
        "combined_attack_block_rate",
        "combined_utility_rate_or",
        "combined_attack_block_rate_or",
        "combined_utility_rate_and",
        "combined_attack_block_rate_and",
        "combined_utility_rate_xor",
        "combined_attack_block_rate_xor",
        "combined_utility_rate_intent",
        "combined_attack_block_rate_intent",
        "threshold",
        "training_utility_rate",
    ]

    for m in percentage_metrics:
        values = [x.get(m, 0.0) for x in results_list]
        agg[f"{m}_mean"] = np.mean(values)
        agg[f"{m}_std"] = np.std(values)

    # Calculate means for count metrics (round to int)
    count_metrics = [
        "training_samples",
        "policy_utility_total",
        "policy_utility_allowed",
        "policy_utility_blocked",
        "policy_attack_total",
        "policy_attack_allowed",
        "policy_attack_blocked",
        "combined_utility_total",
        "combined_utility_allowed",
        "combined_utility_blocked",
        "combined_attack_total",
        "combined_attack_allowed",
        "combined_attack_blocked",
    ]
    
    # Add combined count metrics for different logics
    for suffix in ["_or", "_and", "_xor", "_intent"]:
        count_metrics.extend([
            f"combined_utility_allowed{suffix}",
            f"combined_utility_blocked{suffix}",
            f"combined_attack_allowed{suffix}",
            f"combined_attack_blocked{suffix}",
        ])

    for m in count_metrics:
        values = [x.get(m, 0) for x in results_list]
        agg[f"{m}_mean"] = int(round(np.mean(values)))

    agg["agent"] = agent
    agg["training_fraction"] = fraction
    agg["seeds"] = len(results_list)

    # Save aggregated summary
    summary_path = os.path.join(summary_dir, "metrics.json")
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"    Aggregated results saved to {summary_path}")

    return agg


# ============================================================================
# Main Experiment Function
# ============================================================================


def run_experiment(args):
    """Run the complete security-utility tradeoff experiment with multiple seeds."""

    print("=" * 80)
    print("AGENTDOJO SECURITY-UTILITY TRADEOFF EXPERIMENT")
    print("=" * 80)
    print(f"Dataset: {args.dataset_root}")
    print(f"Intent alignment: {args.intent_root}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Target utility rate: {args.target_utility_rate}%")
    print(f"Combination logic: {args.logic}")
    print(f"Workers: {args.workers}")
    print(f"Seeds: {SEEDS}")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-build cache
    build_input_cache(
        AGENTS, args.dataset_root, args.mode, args.cache_root, args.workers
    )

    # Store all aggregated results for final plotting
    all_aggregated_results = []

    # Run experiment for each agent
    for agent in AGENTS:
        print(f"\n{'='*80}")
        print(f"AGENT: {agent}")
        print(f"{'='*80}")

        # Load all utility and attack files
        all_utility_files = load_utility_traces(agent, args.dataset_root)
        all_attack_files = load_attack_traces(agent, args.dataset_root)

        print(
            f"Found {len(all_utility_files)} utility traces, {len(all_attack_files)} attack traces"
        )

        if not all_utility_files:
            print(f"  ERROR: No utility traces found for {agent}!")
            continue

        # Train policies on different fractions
        for fraction in TRAINING_FRACTIONS:
            print(f"\n  Training Fraction: {int(fraction*100)}%")
            
            # Run experiment for each seed
            seed_results = []
            
            for seed in SEEDS:
                print(f"    Seed {seed}: Training and evaluating...")
                
                # Sample training data with this seed
                training_files = sample_training_fraction(
                    all_utility_files, fraction, seed
                )

                # Train policy
                policy, metadata = train_policy_on_fraction(
                    agent,
                    training_files,
                    all_attack_files,
                    fraction,
                    args.dataset_root,
                    args.output_dir,
                    args.mode,
                    args.workers,
                    args.cache_root,
                    seed,
                    args.target_utility_rate,
                )

                if policy is None:
                    continue

                # Evaluate combined system
                eval_results, utility_scores, attack_scores = evaluate_combined_system(
                    agent,
                    policy,
                    metadata["threshold"],
                    all_utility_files,
                    all_attack_files,
                    args.dataset_root,
                    args.intent_root,
                    args.mode,
                    args.cache_root,
                    args.workers,
                    args.logic,
                )

                # Plot score distribution for this seed
                plot_score_distribution(
                    agent, fraction, utility_scores, attack_scores, args.output_dir, seed
                )

                # Combine metadata and results
                combined_results = {**metadata, **eval_results}
                seed_results.append(combined_results)

                # Save policy for this seed
                policy_dir = os.path.join(
                    args.output_dir, agent, f"train_frac_{int(fraction*100)}", f"seed_{seed}"
                )
                os.makedirs(policy_dir, exist_ok=True)

                policy_path = os.path.join(policy_dir, "policy.json")
                policy.save(policy_path)

                metrics_path = os.path.join(policy_dir, "metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(combined_results, f, indent=2)

                print(f"      Saved policy and metrics to {policy_dir}")

            # Aggregate results across seeds
            aggregated = aggregate_results(seed_results, agent, fraction, args.output_dir)
            if aggregated:
                all_aggregated_results.append(aggregated)

    # Save aggregated summary
    summary_path = os.path.join(args.output_dir, "aggregated_experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_aggregated_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Aggregated summary: {summary_path}")

    # Print combined system results table for 100% training fraction (using aggregated)
    print_combined_system_table(all_aggregated_results)

    # Generate summary plots (using aggregated results)
    generate_plots(all_aggregated_results, args.output_dir)

    # Generate detailed summary table (using aggregated results from files)
    agents = [agent.split('/')[-1] for agent in glob(os.path.join(args.output_dir, '*')) if os.path.isdir(agent) and os.path.basename(agent) in ['banking', 'slack', 'travel', 'workspace']]
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    generate_detailed_summary_table(args.output_dir, agents if agents else ['banking', 'slack', 'travel', 'workspace'], fractions)

    # Generate detailed tradeoff plots (using aggregated results from files)
    generate_detailed_plots(args.output_dir, agents if agents else ['banking', 'slack', 'travel', 'workspace'], fractions)


def generate_plots(results: List[Dict], output_dir: str):
    """Generate summary plots for the experiment using aggregated results."""

    print(f"\nGenerating summary plots from aggregated results...")

    # Organize results by agent and fraction
    agents_data = {}
    for result in results:
        agent = result["agent"]
        if agent not in agents_data:
            agents_data[agent] = {
                "fractions": [],
                "policy_utility": [],
                "policy_utility_std": [],
                "policy_attack": [],
                "policy_attack_std": [],
                "combined_utility_or": [],
                "combined_utility_or_std": [],
                "combined_attack_or": [],
                "combined_attack_or_std": [],
                "combined_utility_and": [],
                "combined_utility_and_std": [],
                "combined_attack_and": [],
                "combined_attack_and_std": [],
            }

        agents_data[agent]["fractions"].append(result["training_fraction"] * 100)
        agents_data[agent]["policy_utility"].append(result.get("policy_utility_rate_mean", 0))
        agents_data[agent]["policy_utility_std"].append(result.get("policy_utility_rate_std", 0))
        agents_data[agent]["policy_attack"].append(result.get("policy_attack_block_rate_mean", 0))
        agents_data[agent]["policy_attack_std"].append(result.get("policy_attack_block_rate_std", 0))

        # Combined OR
        agents_data[agent]["combined_utility_or"].append(
            result.get("combined_utility_rate_or_mean", 0)
        )
        agents_data[agent]["combined_utility_or_std"].append(
            result.get("combined_utility_rate_or_std", 0)
        )
        agents_data[agent]["combined_attack_or"].append(
            result.get("combined_attack_block_rate_or_mean", 0)
        )
        agents_data[agent]["combined_attack_or_std"].append(
            result.get("combined_attack_block_rate_or_std", 0)
        )

        # Combined AND
        agents_data[agent]["combined_utility_and"].append(
            result.get("combined_utility_rate_and_mean", 0)
        )
        agents_data[agent]["combined_utility_and_std"].append(
            result.get("combined_utility_rate_and_std", 0)
        )
        agents_data[agent]["combined_attack_and"].append(
            result.get("combined_attack_block_rate_and_mean", 0)
        )
        agents_data[agent]["combined_attack_and_std"].append(
            result.get("combined_attack_block_rate_and_std", 0)
        )

    # ============================
    # 1. Policy-Only Plot
    # ============================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Security-Utility Tradeoff Across Training Fractions (Mean ± Std)", fontsize=16)

    for agent, data in agents_data.items():
        # Policy Utility
        axes[0].errorbar(
            data["fractions"], data["policy_utility"], 
            yerr=data["policy_utility_std"],
            marker="o", label=agent, capsize=5
        )
        # Policy Attack Blocking
        axes[1].errorbar(
            data["fractions"], data["policy_attack"],
            yerr=data["policy_attack_std"],
            marker="o", label=agent, capsize=5
        )

    axes[0].set_title("Policy-Only: Utility Success Rate")
    axes[0].set_xlabel("Training Fraction (%)")
    axes[0].set_ylabel("Utility Rate (%)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Policy-Only: Attack Blocking Rate")
    axes[1].set_xlabel("Training Fraction (%)")
    axes[1].set_ylabel("Block Rate (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    policy_path = os.path.join(output_dir, "policy_security_utility_tradeoff.png")
    plt.savefig(policy_path)
    plt.close()
    print(f"  Saved plot to: {policy_path}")

    # ============================
    # 2. Combined System (OR) Plot
    # ============================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Security-Utility Tradeoff Across Training Fractions (Mean ± Std)", fontsize=16)

    for agent, data in agents_data.items():
        # Combined Utility
        axes[0].errorbar(
            data["fractions"], data["combined_utility_or"],
            yerr=data["combined_utility_or_std"],
            marker="s", label=agent, capsize=5
        )
        # Combined Attack Blocking
        axes[1].errorbar(
            data["fractions"], data["combined_attack_or"],
            yerr=data["combined_attack_or_std"],
            marker="s", label=agent, capsize=5
        )

    axes[0].set_title("Combined (OR): Utility Success Rate")
    axes[0].set_xlabel("Training Fraction (%)")
    axes[0].set_ylabel("Utility Rate (%)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Combined (OR): Attack Blocking Rate")
    axes[1].set_xlabel("Training Fraction (%)")
    axes[1].set_ylabel("Block Rate (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    combined_or_path = os.path.join(
        output_dir, "combined_or_security_utility_tradeoff.png"
    )
    plt.savefig(combined_or_path)
    plt.close()
    print(f"  Saved plot to: {combined_or_path}")

    # ============================
    # 3. Combined System (AND) Plot
    # ============================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Security-Utility Tradeoff Across Training Fractions (Mean ± Std)", fontsize=16)

    for agent, data in agents_data.items():
        # Combined Utility
        axes[0].errorbar(
            data["fractions"], data["combined_utility_and"],
            yerr=data["combined_utility_and_std"],
            marker="^", label=agent, capsize=5
        )
        # Combined Attack Blocking
        axes[1].errorbar(
            data["fractions"], data["combined_attack_and"],
            yerr=data["combined_attack_and_std"],
            marker="^", label=agent, capsize=5
        )

    axes[0].set_title("Combined (AND): Utility Success Rate")
    axes[0].set_xlabel("Training Fraction (%)")
    axes[0].set_ylabel("Utility Rate (%)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Combined (AND): Attack Blocking Rate")
    axes[1].set_xlabel("Training Fraction (%)")
    axes[1].set_ylabel("Block Rate (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    combined_and_path = os.path.join(
        output_dir, "combined_and_security_utility_tradeoff.png"
    )
    plt.savefig(combined_and_path)
    plt.close()
    print(f"  Saved plot to: {combined_and_path}")


def print_combined_system_table(results: List[Dict]):
    """Print a summary table for combined system results at 100% training fraction (aggregated)."""

    # Filter for 100% training fraction
    results_100 = [r for r in results if r["training_fraction"] == 1.0]

    if not results_100:
        return

    # Sort by agent name
    results_100 = sorted(results_100, key=lambda x: x["agent"])

    print()
    print("=" * 100)
    print(
        "COMBINED SYSTEM RESULTS (Policy + Intent Alignment, Training Fraction: 100%, Aggregated)"
    )
    print("=" * 100)
    print()
    print(
        f"{'Agent':<12} {'Utility %':<20} {'Attack Block %':<20} {'Utility Total':<15} {'Attack Total':<15}"
    )
    print("-" * 100)

    total_utility_mean = 0
    total_utility_std = 0
    total_attack_mean = 0
    total_attack_std = 0

    for r in results_100:
        util_mean = r.get('combined_utility_rate_mean', 0)
        util_std = r.get('combined_utility_rate_std', 0)
        attack_mean = r.get('combined_attack_block_rate_mean', 0)
        attack_std = r.get('combined_attack_block_rate_std', 0)
        util_total = r.get('combined_utility_total_mean', 0)
        attack_total = r.get('combined_attack_total_mean', 0)
        
        print(
            f"{r['agent']:<12} {util_mean:>6.2f}±{util_std:<10.2f} "
            f"{attack_mean:>6.2f}±{attack_std:<11.2f} {util_total:<15} "
            f"{attack_total:<15}"
        )
        total_utility_mean += util_mean
        total_utility_std += util_std
        total_attack_mean += attack_mean
        total_attack_std += attack_std

    avg_utility_mean = total_utility_mean / len(results_100)
    avg_utility_std = total_utility_std / len(results_100)
    avg_attack_mean = total_attack_mean / len(results_100)
    avg_attack_std = total_attack_std / len(results_100)

    print("-" * 100)
    print(f"{'AVERAGE':<12} {avg_utility_mean:>6.2f}±{avg_utility_std:<10.2f} {avg_attack_mean:>6.2f}±{avg_attack_std:<11.2f}")
    print()


def plot_score_distribution(
    agent_name: str,
    fraction: float,
    utility_scores: List[float],
    attack_scores: List[float],
    output_dir: str,
    seed: int = None,
):
    """
    Plot anomaly score distribution for utility vs attack traces.

    Args:
        agent_name: Name of agent
        fraction: Training fraction
        utility_scores: List of utility scores
        attack_scores: List of attack scores
        output_dir: Output directory
        seed: Random seed (optional, for per-seed plots)
    """
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(
        utility_scores,
        bins=50,
        alpha=0.6,
        label=f"Utility (N={len(utility_scores)})",
        color="blue",
        density=True,
    )
    plt.hist(
        attack_scores,
        bins=50,
        alpha=0.6,
        label=f"Attack (N={len(attack_scores)})",
        color="red",
        density=True,
    )

    title = f"Anomaly Score Distribution - {agent_name} (Frac: {fraction})"
    if seed is not None:
        title += f" [Seed: {seed}]"
    plt.title(title)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    if seed is not None:
        save_dir = os.path.join(output_dir, agent_name, f"train_frac_{int(fraction*100)}", f"seed_{seed}")
    else:
        save_dir = os.path.join(output_dir, agent_name, f"train_frac_{int(fraction*100)}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "score_distribution.png")

    plt.savefig(save_path)
    plt.close()


def generate_detailed_summary_table(
    output_dir: str, agents: List[str], fractions: List[float]
):
    """
    Generate summary table and CSV with aggregated results across seeds.

    Args:
        output_dir: Output directory
        agents: List of agent names
        fractions: List of training fractions
    """
    print("\n" + "=" * 80)
    print("DETAILED SUMMARY RESULTS (AGGREGATED ACROSS SEEDS)")
    print("=" * 80)

    # Collect all metrics
    all_results = []

    for agent in agents:
        for fraction in fractions:
            metrics_path = os.path.join(
                output_dir, agent, f"train_frac_{int(fraction*100)}", "aggregated", "metrics.json"
            )

            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    all_results.append(metrics)

    if not all_results:
        print("No aggregated results found!")
        return

    # Print table with policy-only results
    print(
        f"\n{'Agent':<12} {'Frac':<6} {'Train':<7} {'Utility%':<15} {'Attack%':<15} {'Threshold':<15}"
    )
    print("-" * 95)

    for result in sorted(
        all_results, key=lambda x: (x["agent"], x["training_fraction"])
    ):
        util_mean = result.get('policy_utility_rate_mean', 0)
        util_std = result.get('policy_utility_rate_std', 0)
        attack_mean = result.get('policy_attack_block_rate_mean', 0)
        attack_std = result.get('policy_attack_block_rate_std', 0)
        thresh_mean = result.get('threshold_mean', 0)
        thresh_std = result.get('threshold_std', 0)
        training_samples = result.get('training_samples_mean', 0)
        
        print(
            f"{result['agent']:<12} "
            f"{result['training_fraction']*100:>5.0f}% "
            f"{training_samples:>6} "
            f"{util_mean:>6.2f}±{util_std:<5.2f} "
            f"{attack_mean:>6.2f}±{attack_std:<6.2f} "
            f"{thresh_mean:>6.1f}±{thresh_std:<5.1f}"
        )

    # Print table with combined results
    print("\n" + "=" * 80)
    print("COMBINED RESULTS (AGGREGATED ACROSS SEEDS)")
    print("=" * 80)
    print(f"\n{'Agent':<12} {'Frac':<6} {'Train':<7} {'Utility%':<15} {'Attack%':<15}")
    print("-" * 80)

    for result in sorted(
        all_results, key=lambda x: (x["agent"], x["training_fraction"])
    ):
        util_mean = result.get('combined_utility_rate_mean', 0)
        util_std = result.get('combined_utility_rate_std', 0)
        attack_mean = result.get('combined_attack_block_rate_mean', 0)
        attack_std = result.get('combined_attack_block_rate_std', 0)
        training_samples = result.get('training_samples_mean', 0)
        
        print(
            f"{result['agent']:<12} "
            f"{result['training_fraction']*100:>5.0f}% "
            f"{training_samples:>6} "
            f"{util_mean:>6.2f}±{util_std:<5.2f} "
            f"{attack_mean:>6.2f}±{attack_std:<6.2f}"
        )

    # Save CSV with all results
    csv_path = os.path.join(output_dir, "detailed_summary_results.csv")
    if all_results:
        # Reorder columns to put agent, training_fraction, and seeds first for better readability
        priority_fields = ['agent', 'training_fraction', 'seeds', 'training_samples_mean']
        all_fields = list(all_results[0].keys())
        remaining_fields = [f for f in all_fields if f not in priority_fields]
        fieldnames = priority_fields + remaining_fields

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nSaved detailed summary CSV to {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, "detailed_summary_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved detailed summary JSON to {json_path}")


def generate_detailed_plots(output_dir: str, agents: List[str], fractions: List[float]):
    """
    Generate detailed security-utility tradeoff plots from aggregated results.

    Args:
        output_dir: Output directory containing results
        agents: List of agent names
        fractions: List of training fractions
    """
    print("\n" + "=" * 80)
    print("GENERATING DETAILED TRADEOFF PLOTS (AGGREGATED ACROSS SEEDS)")
    print("=" * 80)

    # Collect all metrics
    all_results = []
    for agent in agents:
        for fraction in fractions:
            metrics_path = os.path.join(
                output_dir, agent, f"train_frac_{int(fraction*100)}", "aggregated", "metrics.json"
            )

            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    all_results.append(metrics)

    if not all_results:
        print("No aggregated results found for plotting!")
        return

    # Organize data by agent
    agent_data = {}
    for result in all_results:
        agent = result["agent"]
        if agent not in agent_data:
            agent_data[agent] = {
                "fractions": [],
                "utility_rates_mean": [],
                "utility_rates_std": [],
                "attack_rates_mean": [],
                "attack_rates_std": [],
                "combined_utility_rates_mean": [],
                "combined_utility_rates_std": [],
                "combined_attack_rates_mean": [],
                "combined_attack_rates_std": [],
                "training_samples": [],
            }
        agent_data[agent]["fractions"].append(result["training_fraction"] * 100)
        agent_data[agent]["utility_rates_mean"].append(result.get("policy_utility_rate_mean", 0))
        agent_data[agent]["utility_rates_std"].append(result.get("policy_utility_rate_std", 0))
        agent_data[agent]["attack_rates_mean"].append(result.get("policy_attack_block_rate_mean", 0))
        agent_data[agent]["attack_rates_std"].append(result.get("policy_attack_block_rate_std", 0))
        agent_data[agent]["combined_utility_rates_mean"].append(
            result.get("combined_utility_rate_mean", 0)
        )
        agent_data[agent]["combined_utility_rates_std"].append(
            result.get("combined_utility_rate_std", 0)
        )
        agent_data[agent]["combined_attack_rates_mean"].append(
            result.get("combined_attack_block_rate_mean", 0)
        )
        agent_data[agent]["combined_attack_rates_std"].append(
            result.get("combined_attack_block_rate_std", 0)
        )
        agent_data[agent]["training_samples"].append(result.get("training_samples_mean", 0))

    # Sort by fraction
    for agent in agent_data:
        sorted_indices = sorted(
            range(len(agent_data[agent]["fractions"])),
            key=lambda i: agent_data[agent]["fractions"][i],
        )
        agent_data[agent]["fractions"] = [
            agent_data[agent]["fractions"][i] for i in sorted_indices
        ]
        agent_data[agent]["utility_rates_mean"] = [
            agent_data[agent]["utility_rates_mean"][i] for i in sorted_indices
        ]
        agent_data[agent]["utility_rates_std"] = [
            agent_data[agent]["utility_rates_std"][i] for i in sorted_indices
        ]
        agent_data[agent]["attack_rates_mean"] = [
            agent_data[agent]["attack_rates_mean"][i] for i in sorted_indices
        ]
        agent_data[agent]["attack_rates_std"] = [
            agent_data[agent]["attack_rates_std"][i] for i in sorted_indices
        ]
        agent_data[agent]["combined_utility_rates_mean"] = [
            agent_data[agent]["combined_utility_rates_mean"][i] for i in sorted_indices
        ]
        agent_data[agent]["combined_utility_rates_std"] = [
            agent_data[agent]["combined_utility_rates_std"][i] for i in sorted_indices
        ]
        agent_data[agent]["combined_attack_rates_mean"] = [
            agent_data[agent]["combined_attack_rates_mean"][i] for i in sorted_indices
        ]
        agent_data[agent]["combined_attack_rates_std"] = [
            agent_data[agent]["combined_attack_rates_std"][i] for i in sorted_indices
        ]
        agent_data[agent]["training_samples"] = [
            agent_data[agent]["training_samples"][i] for i in sorted_indices
        ]

    # Create plots
    colors = {
        "banking": "#1f77b4",
        "slack": "#ff7f0e",
        "travel": "#2ca02c",
        "workspace": "#d62728",
    }

    # Plot 1: Policy Utility Success Rate vs Training Fraction (with error bars)
    plt.figure(figsize=(10, 6))
    for agent in sorted(agent_data.keys()):
        data = agent_data[agent]
        plt.errorbar(
            data["fractions"],
            data["utility_rates_mean"],
            yerr=data["utility_rates_std"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=agent.capitalize(),
            color=colors.get(agent, None),
            capsize=5,
        )

    plt.xlabel("Training Data Fraction (%)", fontsize=12)
    plt.ylabel("Utility Success Rate (%)", fontsize=12)
    plt.title(
        "Policy Utility Success Rate vs Training Data Fraction (Mean ± Std)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0, 105)
    plt.tight_layout()

    utility_plot_path = os.path.join(
        output_dir, "detailed_policy_utility_vs_fraction.png"
    )
    plt.savefig(utility_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Policy Attack Blocking Rate vs Training Fraction (with error bars)
    plt.figure(figsize=(10, 6))
    for agent in sorted(agent_data.keys()):
        data = agent_data[agent]
        plt.errorbar(
            data["fractions"],
            data["attack_rates_mean"],
            yerr=data["attack_rates_std"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=agent.capitalize(),
            color=colors.get(agent, None),
            capsize=5,
        )

    plt.xlabel("Training Data Fraction (%)", fontsize=12)
    plt.ylabel("Attack Blocking Rate (%)", fontsize=12)
    plt.title(
        "Policy Attack Blocking Rate vs Training Data Fraction (Mean ± Std)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0, 105)
    plt.tight_layout()

    attack_plot_path = os.path.join(
        output_dir, "detailed_policy_attack_blocking_vs_fraction.png"
    )
    plt.savefig(attack_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Combined Utility Success Rate vs Training Fraction (with error bars)
    plt.figure(figsize=(10, 6))
    for agent in sorted(agent_data.keys()):
        data = agent_data[agent]
        plt.errorbar(
            data["fractions"],
            data["combined_utility_rates_mean"],
            yerr=data["combined_utility_rates_std"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=agent.capitalize(),
            color=colors.get(agent, None),
            capsize=5,
        )

    plt.xlabel("Training Data Fraction (%)", fontsize=12)
    plt.ylabel("Utility Success Rate (%)", fontsize=12)
    plt.title(
        "Combined Utility Success Rate vs Training Data Fraction (Mean ± Std)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0, 105)
    plt.tight_layout()

    combined_utility_plot_path = os.path.join(
        output_dir, "detailed_combined_utility_vs_fraction.png"
    )
    plt.savefig(combined_utility_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: Combined Attack Blocking Rate vs Training Fraction (with error bars)
    plt.figure(figsize=(10, 6))
    for agent in sorted(agent_data.keys()):
        data = agent_data[agent]
        plt.errorbar(
            data["fractions"],
            data["combined_attack_rates_mean"],
            yerr=data["combined_attack_rates_std"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=agent.capitalize(),
            color=colors.get(agent, None),
            capsize=5,
        )

    plt.xlabel("Training Data Fraction (%)", fontsize=12)
    plt.ylabel("Attack Blocking Rate (%)", fontsize=12)
    plt.title(
        "Combined Attack Blocking Rate vs Training Data Fraction (Mean ± Std)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0, 105)
    plt.tight_layout()

    combined_attack_plot_path = os.path.join(
        output_dir, "detailed_combined_attack_blocking_vs_fraction.png"
    )
    plt.savefig(combined_attack_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print("DETAILED PLOTTING COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Security-Utility Tradeoff Experiment for AgentDojo Dataset"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory of AgentDojo dataset (e.g., /path/to/AD_run1/gpt-4o-2024-05-13)",
    )
    parser.add_argument(
        "--intent_root",
        type=str,
        required=True,
        help="Root directory for intent alignment results (e.g., /path/to/agentdojo_intent_alignment_results/AD_run1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for experiment results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "semantic"],
        help="Mode for DFG extraction (default: heuristic)",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default="cache",
        help="Root directory for cache (default: cache)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--target_utility_rate",
        type=float,
        default=90.0,
        help="Target utility rate for threshold selection (default: 90.0)",
    )
    parser.add_argument(
        "--logic",
        type=str,
        default="OR",
        choices=["OR", "AND", "XOR"],
        help="Logic for combining policy and intent (default: OR)",
    )

    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
