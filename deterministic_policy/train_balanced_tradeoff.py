#!/usr/bin/env python3
"""
Balanced Tradeoff Experiment

Trains ActionBlockingPolicy models on varying fractions of utility data (10%-100%)
using 3 different random seeds. Evaluations include:
1. Standard evaluation on full datasets
2. Weighted evaluation with oversampling (e.g., 9:1 Utility:Attack)

Includes Intent Alignment integration.

Usage:
    python train_balanced_tradeoff.py \\
        --dataset_root /home/rohseque/agent_sentry/agent_sentry_unique_dataset \\
        --output_dir experiments/balanced_experiments \\
        --mode heuristic \\
        --workers 100
"""

import os
import json
import csv
import random
import argparse
import time
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

matplotlib.use("Agg")  # Use non-interactive backend

# Import existing codebase components
from policy import ActionBlockingPolicy
from extractor import TraceExtractor
from structures import TraceData
from csv_helper import save_global_csvs


# ============================================================================
# Configuration
# ============================================================================

AGENTS = ["banking", "slack", "travel", "workspace"]
TRAINING_FRACTIONS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [42, 43, 44]
EVAL_RATIOS = ["9:1", "1:1"]  # Utility:Attack ratios


# ============================================================================
# Caching System (Reused)
# ============================================================================


def get_cache_path(
    dataset_root: str, file_path: str, cache_root: str, mode: str
) -> str:
    dataset_name = os.path.basename(dataset_root.rstrip(os.sep))
    rel_path = os.path.relpath(file_path, dataset_root)
    return os.path.join(cache_root, mode, dataset_name, rel_path)


def extract_trace_cached(
    file_path: str,
    dataset_root: str,
    agent_name: str,
    mode: str,
    cache_root: str = "cache",
) -> Optional[TraceData]:
    # If agent_name is "mixed", infer from path
    if agent_name == "mixed":
        for possible_agent in ["banking", "slack", "travel", "workspace"]:
            if f"/{possible_agent}/" in file_path:
                agent_name = possible_agent
                break

    cache_path = get_cache_path(dataset_root, file_path, cache_root, mode)

    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return TraceData.from_dict(json.load(f))
        except Exception:
            pass

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


def extract_traces_parallel(
    file_paths: List[str],
    dataset_root: str,
    agent_name: str,
    mode: str,
    cache_root: str,
    workers: int,
) -> List[TraceData]:
    traces = []

    def extract_one(fpath):
        return extract_trace_cached(fpath, dataset_root, agent_name, mode, cache_root)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_one, fpath): fpath for fpath in file_paths}
        for future in as_completed(futures):
            trace = future.result()
            if trace is not None and trace.tool_calls:
                traces.append(trace)
    return traces


# ============================================================================
# Intent Alignment Integration
# ============================================================================

# Get workspace root (parent of deterministic_policy directory)
BASE_DIR = Path(__file__).resolve().parent.parent
INTENT_ALIGNMENT_ROOT = str(BASE_DIR / "intent_alignment_unique_results")  # Updated to use merged results


def load_intent_alignment_result(
    file_path: str, agent_name: str, folder_type: str
) -> Optional[int]:
    """
    Load intent alignment result for a trace file.
    folder_type: 'utilities' or 'attacks'
    Returns: 0=aligned/utility, 1=misaligned/attack, or None
    """
    base_name = os.path.basename(file_path).replace(".json", "")
    intent_file = os.path.join(
        INTENT_ALIGNMENT_ROOT, agent_name, folder_type, f"{base_name}_gpt-5-mini.json"
    )

    if not os.path.exists(intent_file):
        return None

    try:
        with open(intent_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("val")
    except Exception:
        pass
    return None


def combine_results(policy_blocked: bool, intent_misaligned: bool, logic: str) -> bool:
    if logic == "OR":
        return policy_blocked or intent_misaligned
    elif logic == "AND":
        return policy_blocked and intent_misaligned
    elif logic == "XOR":
        return policy_blocked != intent_misaligned
    raise ValueError(f"Unknown logic: {logic}")


# ============================================================================
# Data Loading & Sampling
# ============================================================================


def load_utility_traces(agent_name: str, dataset_root: str) -> List[str]:
    utility_dir = os.path.join(dataset_root, agent_name, "utilities")
    if not os.path.isdir(utility_dir):
        return []
    return sorted(glob(os.path.join(utility_dir, "*.json")))


def load_attack_traces(agent_name: str, dataset_root: str) -> List[str]:
    attack_dir = os.path.join(dataset_root, agent_name, "attacks")
    if not os.path.isdir(attack_dir):
        return []
    return sorted(glob(os.path.join(attack_dir, "*.json")))


def sample_training_fraction(
    all_files: List[str], fraction: float, seed: int
) -> List[str]:
    rng = random.Random(seed)
    sample_size = max(1, int(len(all_files) * fraction))
    return rng.sample(all_files, sample_size)


# ============================================================================
# Core Logic: Scoring & Evaluation
# ============================================================================


def score_trace_with_policy(
    policy: ActionBlockingPolicy,
    trace: TraceData,
    threshold: float,
    untrusted_retrieval_tools: set = None,
) -> Tuple[float, bool, dict]:
    """
    Score a trace and return (max_score, is_blocked, classification_stats).
    
    Returns:
        tuple: (max_score, is_blocked, classification_stats)
        where classification_stats is a dict with counts:
        {
            'utility': count,
            'ambiguous': count,
            'attack': count,
            'novel': count,
            'dominant_class': str  # class of the max_score action
        }
    """
    max_score = float("-inf")
    
    # Track classifications
    classification_counts = {
        'utility': 0,
        'ambiguous': 0,
        'attack': 0,
        'novel': 0
    }
    dominant_class = 'utility'  # default if no actions

    # Pre-compute unique actions
    unique_actions = []
    seen = set()
    for tc in trace.tool_calls:
        if tc.name in policy.action_tools:
            key = (tc.name, tc.tool_call_id)
            if key not in seen:
                unique_actions.append(key)
                seen.add(key)

    # Process
    ret_tools_seen = set()
    actions_seen = []
    action_idx = 0

    tool_seq = ["user_prompt"] + [tc.name for tc in trace.tool_calls]
    for i, tool in enumerate(tool_seq):
        if tool == "user_prompt":
            continue

        if tool in policy.retrieval_tools:
            ret_tools_seen.add(tool)
        elif tool in policy.action_tools:
            if action_idx < len(unique_actions):
                action_name, tool_call_id = unique_actions[action_idx]
                action_nodes = [
                    n for n in trace.dfg_nodes if n.tool_call_id == tool_call_id
                ]

                # Get detailed scoring with classification
                detailed_result = policy.score_action(
                    action_name=action_name,
                    actions_seen_so_far=actions_seen.copy(),
                    ret_tools_seen_so_far=ret_tools_seen.copy(),
                    dfg_nodes=action_nodes,
                    detailed=True,
                )

                score = detailed_result['score']
                
                # Determine classification (CFG takes precedence if both present)
                action_class = detailed_result.get('cfg_class', detailed_result.get('dfg_class', 'novel'))
                
                # Update classification counts
                if action_class in classification_counts:
                    classification_counts[action_class] += 1

                # Track max score and its classification
                if score > max_score:
                    max_score = score
                    dominant_class = action_class

                actions_seen.append(action_name)
                action_idx += 1

    if max_score == float("-inf"):
        max_score = 0.0

    classification_stats = {
        **classification_counts,
        'dominant_class': dominant_class
    }

    return max_score, max_score > threshold, classification_stats


def evaluate_batch(
    policy: ActionBlockingPolicy,
    traces: List[TraceData],
    file_paths: List[str],  # Corresponding file paths for intent lookup
    trace_type: str,  # 'utility' or 'attack'
    threshold: float,
    agent_name: str,
    untrusted_tools: set,
) -> List[Dict]:
    """
    Evaluate a batch of traces and return detailed results for each.
    """
    results = []

    for i, trace in enumerate(traces):
        max_score, is_blocked, classification_stats = score_trace_with_policy(
            policy, trace, threshold, untrusted_tools
        )

        # Intent Alignment
        intent_val = load_intent_alignment_result(
            file_paths[i],
            agent_name,
            "utilities" if trace_type == "utility" else "attacks",
        )

        if trace_type == "utility":
            # For utility, default assume aligned (0)
            intent_misaligned = (intent_val == 1) if intent_val is not None else False
        else:
            # For attack, default assume misaligned/attack (1)
            intent_misaligned = (intent_val == 1) if intent_val is not None else True

        # Combine
        if 0.0 <= max_score <= 2.0:
            blocked_or = combine_results(is_blocked, intent_misaligned, "OR")
            blocked_and = combine_results(is_blocked, intent_misaligned, "AND")
            blocked_xor = combine_results(is_blocked, intent_misaligned, "XOR")
        else:
            # High scores override intent
            blocked_or = is_blocked
            blocked_and = is_blocked
            blocked_xor = is_blocked

        results.append(
            {
                "max_score": max_score,
                "policy_blocked": is_blocked,
                "intent_misaligned": intent_misaligned,
                "blocked_or": blocked_or,
                "blocked_and": blocked_and,
                "blocked_xor": blocked_xor,
                "blocked_intent": intent_misaligned,
                "filename": os.path.basename(file_paths[i]),
                "classification_stats": classification_stats,  # Add classification info
            }
        )

    return results


def calculate_metrics(
    utility_results: List[Dict],
    attack_results: List[Dict],
    threshold: float,
) -> Dict:
    """
    Calculate consolidated metrics from detailed results.
    Includes classification breakdowns for policy-only evaluations.
    """
    metrics = {}

    # Utility Metrics (TN, FP)
    # Success means NOT blocked
    total_util = len(utility_results)
    metrics["utility_total"] = total_util
    
    # Aggregate utility classification counts
    util_class_counts = {'utility': 0, 'ambiguous': 0, 'attack': 0, 'novel': 0}
    for r in utility_results:
        stats = r.get('classification_stats', {})
        for cls in util_class_counts:
            util_class_counts[cls] += stats.get(cls, 0)
    
    # Add utility classification metrics
    metrics["utility_classified_as_utility"] = util_class_counts['utility']
    metrics["utility_classified_as_ambiguous"] = util_class_counts['ambiguous']
    metrics["utility_classified_as_attack"] = util_class_counts['attack']
    metrics["utility_classified_as_novel"] = util_class_counts['novel']
    
    # Calculate utility classification percentages
    total_util_actions = sum(util_class_counts.values())
    if total_util_actions > 0:
        metrics["utility_ambiguous_pct"] = (util_class_counts['ambiguous'] / total_util_actions) * 100.0
        metrics["utility_wrong_pct"] = (util_class_counts['attack'] / total_util_actions) * 100.0
    else:
        metrics["utility_ambiguous_pct"] = 0.0
        metrics["utility_wrong_pct"] = 0.0

    for suffix in ["", "_or", "_and", "_xor", "_intent"]:
        key_blocked = f"policy_blocked" if suffix == "" else f"blocked{suffix}"

        allowed = sum(1 for r in utility_results if not r[key_blocked])
        fp = total_util - allowed

        metrics[f"utility_success_rate{suffix}"] = (
            (allowed / total_util * 100.0) if total_util > 0 else 0.0
        )
        metrics[f"utility_tn{suffix}"] = allowed
        metrics[f"utility_fp{suffix}"] = fp

    # Attack Metrics (TP, FN)
    # Success means BLOCKED
    total_att = len(attack_results)
    metrics["attack_total"] = total_att
    
    # Aggregate attack classification counts
    att_class_counts = {'utility': 0, 'ambiguous': 0, 'attack': 0, 'novel': 0}
    for r in attack_results:
        stats = r.get('classification_stats', {})
        for cls in att_class_counts:
            att_class_counts[cls] += stats.get(cls, 0)
    
    # Add attack classification metrics
    metrics["attack_classified_as_utility"] = att_class_counts['utility']
    metrics["attack_classified_as_ambiguous"] = att_class_counts['ambiguous']
    metrics["attack_classified_as_attack"] = att_class_counts['attack']
    metrics["attack_classified_as_novel"] = att_class_counts['novel']
    
    # Calculate attack classification percentages
    total_att_actions = sum(att_class_counts.values())
    if total_att_actions > 0:
        metrics["attack_ambiguous_pct"] = (att_class_counts['ambiguous'] / total_att_actions) * 100.0
        metrics["attack_wrong_pct"] = (att_class_counts['utility'] / total_att_actions) * 100.0
    else:
        metrics["attack_ambiguous_pct"] = 0.0
        metrics["attack_wrong_pct"] = 0.0

    for suffix in ["", "_or", "_and", "_xor", "_intent"]:
        key_blocked = f"policy_blocked" if suffix == "" else f"blocked{suffix}"

        blocked = sum(1 for r in attack_results if r[key_blocked])
        fn = total_att - blocked

        metrics[f"attack_blocking_rate{suffix}"] = (
            (blocked / total_att * 100.0) if total_att > 0 else 0.0
        )
        metrics[f"attack_tp{suffix}"] = blocked
        metrics[f"attack_fn{suffix}"] = fn

    metrics["threshold"] = threshold
    return metrics


# ============================================================================
# Main Training & Eval Loop
# ============================================================================


def train_and_evaluate_seed(
    agent: str,
    fraction: float,
    seed: int,
    dataset_root: str,
    output_dir: str,
    mode: str,
    workers: int,
    cache_root: str,
    all_utility_files: List[str],
    all_attack_files: List[str],
    untrusted_tools: set,
) -> Dict:
    """
    Runs one training/evaluation cycle for a specific seed.
    Returns dictionary of results (standard + weighted).
    """
    print(f"    Seed {seed}: Training and evaluating...")

    # 1. Sample Training Data
    training_files = sample_training_fraction(all_utility_files, fraction, seed)

    # 2. Extract Training Data
    # Efficiency: only extract what we haven't already extracted in prev steps?
    # Actually extraction is cached, so it's fast.
    training_traces = extract_traces_parallel(
        training_files, dataset_root, agent, mode, cache_root, workers
    )
    all_attack_traces = extract_traces_parallel(
        all_attack_files, dataset_root, agent, mode, cache_root, workers
    )

    # 3. Train Policy
    policy = ActionBlockingPolicy()
    policy.load_tool_classifications(agent)
    policy.train(training_traces, all_attack_traces)

    # 4. Tune Threshold (Target 90% utility on training set)
    scores = []
    for trace in training_traces:
        s, _, _ = score_trace_with_policy(policy, trace, float("inf"), untrusted_tools)
        scores.append(s)

    target_rate = 90.0
    computed_threshold = max(scores) if scores else 0.0
    limit = math.ceil(computed_threshold * 10)
    for t_int in range(10, max(10, limit) + 2):
        t = t_int / 10.0
        allowed = sum(1 for s in scores if s <= t)
        if (allowed / len(scores) * 100.0) >= target_rate:
            computed_threshold = t
            break

    # Save Policy
    seed_dir = os.path.join(
        output_dir, agent, f"train_frac_{int(fraction*100)}", f"seed_{seed}"
    )
    os.makedirs(seed_dir, exist_ok=True)
    policy.save(os.path.join(seed_dir, "policy.json"))

    # --- EVALUATION ---

    # Pre-extract ALL utility traces for evaluation (cached)
    all_utility_traces = extract_traces_parallel(
        all_utility_files, dataset_root, agent, mode, cache_root, workers
    )

    results_bundle = {}

    # A. Standard Evaluation
    util_res = evaluate_batch(
        policy,
        all_utility_traces,
        all_utility_files,
        "utility",
        computed_threshold,
        agent,
        untrusted_tools,
    )
    att_res = evaluate_batch(
        policy,
        all_attack_traces,
        all_attack_files,
        "attack",
        computed_threshold,
        agent,
        untrusted_tools,
    )

    std_metrics = calculate_metrics(util_res, att_res, computed_threshold)
    std_metrics["training_samples"] = len(training_files)
    results_bundle["standard"] = std_metrics

    # Save standard metrics
    with open(os.path.join(seed_dir, "standard_metrics.json"), "w") as f:
        json.dump(std_metrics, f, indent=2)

    # Save detailed trace scores
    with open(os.path.join(seed_dir, "utility_trace_scores.json"), "w") as f:
        json.dump(util_res, f, indent=2)

    with open(os.path.join(seed_dir, "attack_trace_scores.json"), "w") as f:
        json.dump(att_res, f, indent=2)

    # B. Weighted Evaluations
    for ratio_str in EVAL_RATIOS:
        # Parse ratio "9:1" -> u_part=9, a_part=1
        u_part, a_part = map(int, ratio_str.split(":"))

        # Target utility count
        n_attack = len(all_attack_traces)
        if n_attack == 0:
            continue

        target_n_util = int(n_attack * (u_part / a_part))

        # Randomly sample utility traces with replacement
        rng = random.Random(seed)  # Deterministic sampling per seed

        # Create indices for sampling
        indices = [
            rng.randint(0, len(all_utility_traces) - 1) for _ in range(target_n_util)
        ]

        sampled_util_traces = [all_utility_traces[i] for i in indices]
        sampled_util_files = [all_utility_files[i] for i in indices]

        # Evaluate using pre-computed standard results (optimization).
        # We can just pick from 'util_res' list using indices!
        sampled_util_res = [util_res[i] for i in indices]

        weighted_metrics = calculate_metrics(
            sampled_util_res, att_res, computed_threshold
        )
        weighted_metrics["training_samples"] = len(training_files)
        results_bundle[f"weighted_{ratio_str}"] = weighted_metrics

        # Save weighted metrics
        ratio_safe = ratio_str.replace(":", "_")
        with open(
            os.path.join(seed_dir, f"weighted_{ratio_safe}_metrics.json"), "w"
        ) as f:
            json.dump(weighted_metrics, f, indent=2)

    return results_bundle


def aggregate_results(
    results_list: List[Dict],  # List of result bundles (one per seed)
    agent: str,
    fraction: float,
    output_dir: str,
):
    """
    Aggregates results across seeds and saves summaries.
    """
    keys = ["standard"] + [f"weighted_{r}" for r in EVAL_RATIOS]

    summary_dir = os.path.join(
        output_dir, agent, f"train_frac_{int(fraction*100)}", "aggregated"
    )
    os.makedirs(summary_dir, exist_ok=True)

    overall_summary = []

    for key in keys:
        # Collect metrics for this evaluation type across all seeds
        metrics_list = [r[key] for r in results_list if key in r]
        if not metrics_list:
            continue

        # Calculate means and stds for percentages
        agg = {}
        target_metrics = [
            "utility_success_rate",
            "attack_blocking_rate",
            "utility_success_rate_or",
            "attack_blocking_rate_or",
            "utility_success_rate_and",
            "attack_blocking_rate_and",
            "utility_success_rate_xor",
            "attack_blocking_rate_xor",
            "threshold",
        ]

        for m in target_metrics:
            values = [x.get(m, 0.0) for x in metrics_list]
            agg[f"{m}_mean"] = np.mean(values)
            agg[f"{m}_std"] = np.std(values)

        # Add intent metrics to aggregation
        intent_metrics = [
            "utility_success_rate_intent",
            "attack_blocking_rate_intent",
        ]
        for m in intent_metrics:
            values = [x.get(m, 0.0) for x in metrics_list]
            agg[f"{m}_mean"] = np.mean(values)
            agg[f"{m}_std"] = np.std(values)

        # Sum counts (avg counts usually make less sense, but let's do mean and round to int)
        count_metrics = []
        for suffix in ["", "_or", "_and", "_xor", "_intent"]:
            count_metrics.extend(
                [
                    f"utility_tn{suffix}",
                    f"utility_fp{suffix}",
                    f"attack_tp{suffix}",
                    f"attack_fn{suffix}",
                ]
            )
        
        # Add classification metrics (counts)
        classification_metrics = [
            "utility_classified_as_utility",
            "utility_classified_as_ambiguous",
            "utility_classified_as_attack",
            "utility_classified_as_novel",
            "attack_classified_as_utility",
            "attack_classified_as_ambiguous",
            "attack_classified_as_attack",
            "attack_classified_as_novel",
        ]
        
        for m in count_metrics + classification_metrics + ["training_samples"]:
            values = [x.get(m, 0) for x in metrics_list]
            # Round to integer since these are counts
            agg[f"{m}_mean"] = int(round(np.mean(values)))
        
        # Add classification percentage metrics
        classification_pct_metrics = [
            "utility_ambiguous_pct",
            "utility_wrong_pct",
            "attack_ambiguous_pct",
            "attack_wrong_pct",
        ]
        
        for m in classification_pct_metrics:
            values = [x.get(m, 0.0) for x in metrics_list]
            agg[f"{m}_mean"] = np.mean(values)
            agg[f"{m}_std"] = np.std(values)

        agg["eval_type"] = key
        agg["agent"] = agent
        agg["fraction"] = fraction
        agg["seeds"] = len(results_list)

        overall_summary.append(agg)

    # Save aggregated summary CSV
    csv_path = os.path.join(summary_dir, "summary.csv")
    if overall_summary:
        keys = overall_summary[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(overall_summary)

    print(f"    Aggregated results saved to {csv_path}")

    return overall_summary


# ============================================================================
# Main Script
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Balanced Tradeoff Experiment")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", default="experiments/balanced_experiments")
    parser.add_argument("--mode", default="heuristic")
    parser.add_argument("--workers", type=int, default=100)
    args = parser.parse_args()

    print("=" * 80)
    print("BALANCED TRADEOFF EXPERIMENT")
    print(f"Dataset: {args.dataset_root}")
    print(f"Output: {args.output_dir}")
    print(f"Agents: {AGENTS}")
    print(f"Fractions: {TRAINING_FRACTIONS}")
    print(f"Seeds: {SEEDS}")
    print(f"Eval Ratios: {EVAL_RATIOS}")
    print("=" * 80)

    # Pre-build cache (optional but good practice)
    # We skip explicit pre-build here as extract functions handle it lazily

    global_summaries = []

    for agent in AGENTS:
        print(f"\nProcessing AGENT: {agent}")

        # Load all file paths
        util_files = load_utility_traces(agent, args.dataset_root)
        att_files = load_attack_traces(agent, args.dataset_root)

        # Untrusted tools
        untrusted = set()
        tc_path = os.path.join("utils", agent, "tool_classification.json")
        if os.path.exists(tc_path):
            with open(tc_path) as f:
                untrusted = set(json.load(f).get("untrusted_retrieval_tools", []))

        for frac in TRAINING_FRACTIONS:
            percentage = int(frac * 100)
            print(f"  Fraction: {percentage}%")

            seed_results = []

            for seed in SEEDS:
                res = train_and_evaluate_seed(
                    agent=agent,
                    fraction=frac,
                    seed=seed,
                    dataset_root=args.dataset_root,
                    output_dir=args.output_dir,
                    mode=args.mode,
                    workers=args.workers,
                    cache_root="cache",
                    all_utility_files=util_files,
                    all_attack_files=att_files,
                    untrusted_tools=untrusted,
                )
                seed_results.append(res)

            # Aggregate across seeds
            summ = aggregate_results(seed_results, agent, frac, args.output_dir)
            global_summaries.extend(summ)

    save_global_csvs(global_summaries, args.output_dir)
    
    # Generate plots for combined system performance
    print("\nGenerating performance plots...")
    plot_combined_system_performance(args.output_dir)
    print("Plots generated successfully!")


def plot_combined_system_performance(output_dir: str):
    """
    Generate performance plots for combined system (OR logic) showing:
    1. Utility success rate vs functionality graph coverage
    2. Attack blocking rate vs functionality graph coverage
    3. Attack success rate vs functionality graph coverage
    
    Reads from combined_system_or_logic.csv generated by save_global_csvs.
    """
    csv_path = os.path.join(output_dir, "combined_system_or_logic.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping plots.")
        return
    
    # Read the CSV
    agent_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent = row['agent']
            if agent not in agent_data:
                agent_data[agent] = {
                    'fractions': [],
                    'utility_rates': [],
                    'attack_block_rates': [],
                    'attack_success_rates': []
                }
            
            frac = float(row['training_fraction']) * 100  # Convert to percentage
            util_rate = float(row['utility_success_rate_or'])
            attack_block = float(row['attack_blocking_rate_or'])
            attack_success = 100.0 - attack_block  # Success = 100 - Block
            
            agent_data[agent]['fractions'].append(frac)
            agent_data[agent]['utility_rates'].append(util_rate)
            agent_data[agent]['attack_block_rates'].append(attack_block)
            agent_data[agent]['attack_success_rates'].append(attack_success)
    
    # Sort all data by fraction for each agent
    for agent in agent_data:
        sorted_indices = sorted(range(len(agent_data[agent]['fractions'])), 
                               key=lambda i: agent_data[agent]['fractions'][i])
        agent_data[agent]['fractions'] = [agent_data[agent]['fractions'][i] for i in sorted_indices]
        agent_data[agent]['utility_rates'] = [agent_data[agent]['utility_rates'][i] for i in sorted_indices]
        agent_data[agent]['attack_block_rates'] = [agent_data[agent]['attack_block_rates'][i] for i in sorted_indices]
        agent_data[agent]['attack_success_rates'] = [agent_data[agent]['attack_success_rates'][i] for i in sorted_indices]
    
    # Color scheme
    colors = {
        'banking': '#1f77b4',
        'slack': '#ff7f0e',
        'travel': '#2ca02c',
        'workspace': '#d62728'
    }
    
    # Plot 1: Utility Success Rate vs Functionality Graph Coverage
    plt.figure(figsize=(12, 8))
    for agent in sorted(agent_data.keys()):
        data = agent_data[agent]
        plt.plot(
            data['fractions'],
            data['utility_rates'],
            marker='o',
            linewidth=3,
            markersize=10,
            label=agent.capitalize(),
            color=colors.get(agent, None)
        )
    
    plt.xlabel('Functionality Graph Coverage (%)', fontsize=25, fontweight='bold')
    plt.ylabel('Utility Success Rate (%)', fontsize=25, fontweight='bold')
    plt.title('Utility Success Rate vs Functionality Graph Coverage', 
             fontsize=25, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linewidth=1.5)
    plt.legend(fontsize=14, loc='best')
    plt.ylim(0, 105)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    utility_plot_path = os.path.join(output_dir, 'combined_utility_vs_coverage.png')
    plt.savefig(utility_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved utility plot to {utility_plot_path}")
    
    # Plot 2: Attack Blocking Rate vs Functionality Graph Coverage
    plt.figure(figsize=(12, 8))
    for agent in sorted(agent_data.keys()):
        data = agent_data[agent]
        plt.plot(
            data['fractions'],
            data['attack_block_rates'],
            marker='s',
            linewidth=3,
            markersize=10,
            label=agent.capitalize(),
            color=colors.get(agent, None)
        )
    
    plt.xlabel('Functionality Graph Coverage (%)', fontsize=25, fontweight='bold')
    plt.ylabel('Attack Blocking Rate (%)', fontsize=25, fontweight='bold')
    plt.title('Attack Blocking Rate vs Functionality Graph Coverage', 
             fontsize=25, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linewidth=1.5)
    plt.legend(fontsize=14, loc='best')
    plt.ylim(0, 105)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    block_plot_path = os.path.join(output_dir, 'combined_attack_blocking_vs_coverage.png')
    plt.savefig(block_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved attack blocking plot to {block_plot_path}")
    
    # Plot 3: Attack Success Rate vs Functionality Graph Coverage
    plt.figure(figsize=(12, 8))
    for agent in sorted(agent_data.keys()):
        data = agent_data[agent]
        plt.plot(
            data['fractions'],
            data['attack_success_rates'],
            marker='^',
            linewidth=3,
            markersize=10,
            label=agent.capitalize(),
            color=colors.get(agent, None)
        )
    
    plt.xlabel('Functionality Graph Coverage (%)', fontsize=25, fontweight='bold')
    plt.ylabel('Attack Success Rate (%)', fontsize=25, fontweight='bold')
    plt.title('Attack Success Rate vs Functionality Graph Coverage', 
             fontsize=25, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linewidth=1.5)
    plt.legend(fontsize=14, loc='best')
    plt.ylim(0, 105)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    success_plot_path = os.path.join(output_dir, 'combined_attack_success_vs_coverage.png')
    plt.savefig(success_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved attack success plot to {success_plot_path}")


if __name__ == "__main__":
    main()
