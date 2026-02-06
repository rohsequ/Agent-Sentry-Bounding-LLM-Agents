"""
Intent Alignment - Missing Files Processor

This script identifies and processes missing intent alignment results by:
1. Comparing dataset files in agent_sentry_clean_unique_dataset/ against
   existing results in intent_alignment_unique_results/
2. Identifying which files are missing (i.e., haven't been processed yet)
3. Running drift analysis only on those missing files
4. Saving results to intent_alignment_unique_missing/

The script processes files in batches to handle large datasets robustly and
provides detailed progress reporting throughout execution.

Usage:
    python run_intent_alignment_full.py

Configuration:
    - MODEL: The model to use for drift analysis (default: gpt-5-mini)
    - BATCH_SIZE: Number of files to process per batch (default: 500)
    - Paths are configurable via constants at the top of the script
"""

import os
import subprocess
import shutil
import time
import glob
from pathlib import Path

# Configuration
PYTHON_EXECUTABLE = "python3"
# Get workspace root (parent directory of this script)
BASE_DIR = Path(__file__).resolve().parent
INTENT_ALIGNMENT_DIR = BASE_DIR / "intent_alignment"
DATASET_ROOT = BASE_DIR / "agent_sentry_clean_unique_dataset"
EXISTING_RESULTS_DIR = BASE_DIR / "intent_alignment_unique_results"
FINAL_OUTPUT_DIR = BASE_DIR / "intent_alignment_unique_missing"
MODEL = "gpt-5-mini"


def get_missing_files():
    """
    Compare dataset files against existing results to find missing ones.
    Dataset files: agent_sentry_clean_unique_dataset/{agent}/{category}/file.json
    Result files: intent_alignment_unique_results/{agent}/{category}/file_gpt-5-mini.json
    """
    print("Scanning for missing results...")
    missing_files = []

    # Iterate through all source files in the dataset
    for agent_dir in DATASET_ROOT.iterdir():
        if not agent_dir.is_dir():
            continue
        agent_name = agent_dir.name

        for category in ["utilities", "attacks"]:
            source_dir = agent_dir / category
            if not source_dir.exists():
                continue

            # Create destination directory for missing results
            dest_dir = FINAL_OUTPUT_DIR / agent_name / category
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Existing results folder
            existing_dir = EXISTING_RESULTS_DIR / agent_name / category

            for json_file in source_dir.glob("*.json"):
                # Expected output filename: original_name_MODEL.json
                base_name = json_file.stem  # e.g. "claude-3-5-sonnet-20240620_user_task_15_banking_false_none"
                expected_filename = f"{base_name}_{MODEL}.json"

                # Check if result exists in existing results directory
                existing_result = existing_dir / expected_filename if existing_dir.exists() else None
                
                # Check if result exists in the missing/new output directory
                new_result = dest_dir / expected_filename

                # If it exists in NEITHER location, it is missing
                if not (existing_result and existing_result.exists()) and not new_result.exists():
                    missing_files.append(str(json_file))

    return missing_files


def run_alignment():
    """
    Main function to run drift analysis on missing intent alignment results.
    
    This script:
    1. Compares dataset files against existing results
    2. Identifies missing files
    3. Runs drift analysis only on missing files
    4. Moves results to the final output directory
    """
    print("=" * 80)
    print("Intent Alignment - Missing Files Analysis")
    print("=" * 80)
    print(f"Dataset:          {DATASET_ROOT}")
    print(f"Existing Results: {EXISTING_RESULTS_DIR}")
    print(f"Missing Output:   {FINAL_OUTPUT_DIR}")
    print(f"Model:            {MODEL}")
    print("=" * 80)

    # 1. Identify missing files
    missing_files = get_missing_files()
    count = len(missing_files)
    print(f"\n✓ Scan complete: Found {count} missing files to process.")

    if count == 0:
        print("✓ All files have been processed! Nothing to do.")
        return

    # Show distribution by agent and category
    print("\nDistribution of missing files:")
    from collections import Counter
    distribution = Counter()
    for filepath in missing_files:
        path = Path(filepath)
        agent = path.parts[-3]
        category = path.parts[-2]
        distribution[f"{agent}/{category}"] += 1
    
    for key, val in sorted(distribution.items()):
        print(f"  {key}: {val} files")

    # Batch processing for robustness
    BATCH_SIZE = 500
    total_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n→ Will process in {total_batches} batch(es) of up to {BATCH_SIZE} files each")

    # Needs to run from intent_alignment dir for imports to work correctly
    os.chdir(INTENT_ALIGNMENT_DIR)
    print(f"→ Working directory: {os.getcwd()}")

    total_moved = 0
    for i in range(0, count, BATCH_SIZE):
        batch = missing_files[i : i + BATCH_SIZE]
        current_batch_num = i // BATCH_SIZE + 1
        print(f"\n{'=' * 80}")
        print(
            f"Batch {current_batch_num}/{total_batches}: Processing {len(batch)} files..."
        )
        print(f"{'=' * 80}")

        cmd = (
            [
                PYTHON_EXECUTABLE,
                "drift_defense/run_drift_evaluation.py",
                "--model",
                MODEL,
                "--agent_sentry_dataset",
                "--dataset_name",
                "agent_sentry_clean_unique_dataset",
                "--inputs",
            ]
            + batch
            + ["--cot"]
        )

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Batch {current_batch_num} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error in batch {current_batch_num}: {e}")
            print(f"  Continuing with next batch...")
            continue

        # Move results after each batch to verify progress
        print(f"\n→ Moving batch {current_batch_num} results...")
        dataset_name = DATASET_ROOT.name
        generated_results_root = (
            INTENT_ALIGNMENT_DIR / f"{dataset_name}_drift_results" / MODEL / "cot"
        )

        batch_moved = 0
        for category in ["utilities", "attacks"]:
            cat_dir = generated_results_root / category
            if not cat_dir.exists():
                continue

            for agent_dir in cat_dir.iterdir():
                if not agent_dir.is_dir():
                    continue
                agent_name = agent_dir.name
                dest_dir = FINAL_OUTPUT_DIR / agent_name / category
                dest_dir.mkdir(parents=True, exist_ok=True)

                moved_count = 0
                for json_file in agent_dir.glob("*.json"):
                    shutil.move(str(json_file), str(dest_dir / json_file.name))
                    moved_count += 1

                if moved_count > 0:
                    print(f"  ✓ Moved {moved_count} files to {agent_name}/{category}")
                    batch_moved += moved_count

        total_moved += batch_moved
        print(f"✓ Batch {current_batch_num}: Moved {batch_moved} result files")

    print(f"\n{'=' * 80}")
    print(f"✓ All done! Processed {count} files, moved {total_moved} results")
    print(f"✓ Results saved to: {FINAL_OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_alignment()
