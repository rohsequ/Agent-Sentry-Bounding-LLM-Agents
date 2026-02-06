#!/usr/bin/env python3
"""
Run intent alignment (drift evaluation) on AgentDojo datasets.

This script processes the three AgentDojo run directories (AD_run1, AD_run2, AD_run3)
and generates intent alignment results compatible with evaluate_ad.py.

Usage:
    python run_agentdojo_intent_alignment.py --model gpt-5-nano --mode cot
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run intent alignment on AgentDojo datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        choices=[
            "gpt-5-nano", "gpt-5-mini", "gpt-4o-mini",
            "llama3.3", "mistral:7b", "gemma3:12b"
        ],
        help="Model to use for intent alignment (default: gpt-5-nano)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cot",
        choices=["default", "cot", "performance"],
        help="Prompt mode to use (default: cot)"
    )
    parser.add_argument(
        "--ad_dirs",
        type=str,
        nargs="+",
        default=["AD_run1", "AD_run2", "AD_run3"],
        help="AgentDojo directories to process (default: AD_run1 AD_run2 AD_run3)"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="agentdojo_intent_alignment_results",
        help="Output root directory (default: agentdojo_intent_alignment_results)"
    )
    
    args = parser.parse_args()
    
    # Get workspace root (parent directory of this script)
    workspace_root = Path(__file__).resolve().parent
    intent_alignment_dir = workspace_root / "intent_alignment"
    run_drift_script = intent_alignment_dir / "drift_defense" / "run_drift_evaluation.py"
    
    if not run_drift_script.exists():
        print(f"Error: run_drift_evaluation.py not found at {run_drift_script}")
        return 1
    
    # Build list of commands to run
    commands = []
    
    for ad_dir in args.ad_dirs:
        ad_path = workspace_root / ad_dir
        if not ad_path.exists():
            print(f"Warning: AgentDojo directory not found: {ad_path}")
            continue
        
        # Find the model subdirectory (e.g., gpt-4o-2024-05-13)
        model_dirs = [d for d in ad_path.iterdir() if d.is_dir()]
        if not model_dirs:
            print(f"Warning: No model directories found in {ad_path}")
            continue
        
        # Use the first model directory found
        model_dir = model_dirs[0]
        
        # Construct output directory name - separate folder for each dataset
        # The drift script will create: evaluated_traces/{model}/
        # We want this to be under output_root/{ad_dir}, so each dataset has its own folder
        output_dir = workspace_root / args.output_root / ad_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            sys.executable,
            str(run_drift_script),
            "--model", args.model,
            "--agentdojo_format",
            "--security_true",
            "--utility_true",
            "--inputs", str(model_dir),
        ]
        
        # Add prompt mode flag
        if args.mode == "cot":
            cmd.append("--cot")
        elif args.mode == "performance":
            cmd.append("--performance")
        
        # Create log filename
        log_file = output_dir / f"{ad_dir}_{args.model.replace(':', '_')}_{args.mode}.log"
        
        commands.append({
            "cmd": cmd,
            "log_file": log_file,
            "ad_dir": ad_dir,
            "model_dir": model_dir,
            "output_dir": output_dir,
        })
    
    if not commands:
        print("Error: No valid AgentDojo directories found to process")
        return 1
    
    print("="*80)
    print("AGENTDOJO INTENT ALIGNMENT")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Output root: {args.output_root}")
    print(f"\nProcessing {len(commands)} AgentDojo directories:")
    for cmd_info in commands:
        print(f"  - {cmd_info['ad_dir']}: {cmd_info['model_dir']}")
    print()
    
    # Process each dataset sequentially
    for i, cmd_info in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] Processing {cmd_info['ad_dir']}...")
        print(f"Command: {' '.join(cmd_info['cmd'])}")
        print(f"Log file: {cmd_info['log_file']}")
        print(f"Output will be in: {cmd_info['output_dir']}/evaluated_traces/{args.model}/")
        print()
        
        with open(cmd_info['log_file'], 'w') as log_f:
            result = subprocess.run(
                cmd_info['cmd'],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(cmd_info['output_dir'])
            )
        
        if result.returncode == 0:
            print(f"✓ Completed {cmd_info['ad_dir']}")
        else:
            print(f"✗ Failed {cmd_info['ad_dir']} (exit code: {result.returncode})")
            print(f"  Check log: {cmd_info['log_file']}")
    
    print("\n" + "="*80)
    print("INTENT ALIGNMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved in separate folders:")
    for cmd_info in commands:
        result_path = workspace_root / args.output_root / cmd_info['ad_dir'] / 'evaluated_traces' / args.model
        print(f"  {cmd_info['ad_dir']}: {result_path}")
    print("\nNext steps:")
    print(f"  1. Results are in {args.output_root}/{{dataset}}/evaluated_traces/{{model}}/")
    print("  2. Update INTENT_ALIGNMENT_ROOT in evaluate_ad.py to point to this location")
    print("  3. Re-run evaluate_ad.py to see combined system performance")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
