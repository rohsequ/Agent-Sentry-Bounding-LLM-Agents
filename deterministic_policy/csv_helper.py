import os
import csv


def save_global_csvs(global_summaries, output_dir):
    """
    Saves aggregated results into 3 sets of 3 CSVs (OR, Policy, Intent).
    """
    print("\nSaving global summary CSVs...")

    # Group by eval_type
    grouped = {}
    for item in global_summaries:
        etype = item["eval_type"]
        if etype not in grouped:
            grouped[etype] = []
        grouped[etype].append(item)

    # Logic types to save
    # (name_suffix, metric_suffix)
    # Policy: suffix="", Intent: suffix="_intent", OR: suffix="_or"

    csv_configs = [
        ("policy_only", ""),
        ("combined_system_or_logic", "_or"),
        ("intent_only", "_intent"),
        ("classification_analysis", ""),  # New classification analysis CSV
    ]

    # Common columns always needed
    base_cols = ["agent", "fraction", "training_samples_mean"]

    # Metrics to extract for each specific CSV type
    # We map from aggregated key (e.g. utility_success_rate_or_mean) to final csv key (utility_success_rate_or)
    metric_bases = [
        "utility_success_rate",
        "utility_tn",
        "utility_fp",
        "attack_blocking_rate",
        "attack_tp",
        "attack_fn",
    ]
    
    # Classification metrics (only for policy_only)
    classification_bases = [
        "utility_classified_as_utility",
        "utility_classified_as_ambiguous", 
        "utility_classified_as_attack",
        "utility_classified_as_novel",
        "attack_classified_as_utility",
        "attack_classified_as_ambiguous",
        "attack_classified_as_attack",
        "attack_classified_as_novel",
    ]

    for etype, items in grouped.items():
        # Determine base filename prefix
        if etype == "standard":
            prefix = ""
        else:
            # e.g. weighted_9:1 -> weighted_9_1_
            prefix = etype.replace(":", "_") + "_"

        for name_suffix, metric_suffix in csv_configs:
            filename = f"{prefix}{name_suffix}.csv"
            path = os.path.join(output_dir, filename)

            rows = []

            # Use the first item to determine header (conditionally) if needed
            # But we define schema strictly based on constraints

            # Construct header map
            # e.g. "training_samples_mean" -> "training_samples"
            # "utility_success_rate_or_mean" -> "utility_success_rate_or"

            header_map = {
                "agent": "agent",
                "fraction": "training_fraction",
                "training_samples_mean": "training_samples",
                "threshold_mean": "threshold",
            }

            # Add dynamic metrics to map
            for base in metric_bases:
                # Key in aggregation data: {base}{metric_suffix}_mean
                # Key in CSV: {base}{metric_suffix}

                # Special handling: if policy (suffix=""), base is just metric name
                if metric_suffix == "":
                    agg_key = f"{base}_mean"  # e.g. utility_success_rate_mean
                    csv_key = f"{base}"
                else:
                    agg_key = f"{base}{metric_suffix}_mean"
                    csv_key = f"{base}{metric_suffix}"

                header_map[agg_key] = csv_key
            
            # Add classification metrics only for policy_only
            if name_suffix == "policy_only":
                for base in classification_bases:
                    agg_key = f"{base}_mean"
                    csv_key = base
                    header_map[agg_key] = csv_key
            
            # Add classification analysis metrics for classification_analysis CSV
            if name_suffix == "classification_analysis":
                classification_analysis_metrics = [
                    "utility_ambiguous_pct",
                    "utility_wrong_pct",
                    "attack_ambiguous_pct",
                    "attack_wrong_pct",
                ]
                for base in classification_analysis_metrics:
                    agg_key = f"{base}_mean"
                    csv_key = base
                    header_map[agg_key] = csv_key
                    # Also add std columns
                    agg_key_std = f"{base}_std"
                    csv_key_std = f"{base}_std"
                    header_map[agg_key_std] = csv_key_std

            # Build rows
            valid_keys = list(header_map.keys())
            csv_headers = list(header_map.values())

            for item in items:
                row = {}
                for agg_k, csv_k in header_map.items():
                    val = item.get(agg_k, 0)
                    
                    # Round count metrics to integers (TP, TN, FP, FN, training_samples)
                    if any(x in csv_k for x in ["_tp", "_tn", "_fp", "_fn", "training_samples"]):
                        if isinstance(val, (int, float)):
                            val = int(round(val))
                    
                    row[csv_k] = val
                rows.append(row)

            # Sort rows by agent, then fraction
            rows.sort(key=lambda x: (x["agent"], x["training_fraction"]))

            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(rows)

            print(f"  Saved {filename}")
