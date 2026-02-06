import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def parse_filename_metadata(file):
    parts = file.split('_')
    name_lower = file.lower()
    
    dataset_model = "unknown"
    agent_name = "unknown"
    is_utility = False
    task_id = "unknown"
    
    if file.startswith("generated_utility_traces"):
        is_utility = True
        # Format: generated_utility_traces_[AGENT]_[DATASET_MODEL]..._user_task_[ID]...
        # Heuristic utility file
        if len(parts) >= 5:
            agent_name = parts[3]
            try:
                # Find 'user' to locate task part
                user_idx = -1
                for i, part in enumerate(parts):
                    if part == 'user' and (i+1 < len(parts)) and parts[i+1] == 'task':
                        user_idx = i
                        break
                
                if user_idx != -1:
                    # Dataset model is between agent (3) and user (user_idx)
                    # Note: index 3 is agent. parts[4:user_idx] is model.
                    if user_idx > 4:
                        dataset_model = "_".join(parts[4:user_idx])
                    else:
                        dataset_model = parts[4] # Fallback if tight packing?
                    
                    if user_idx + 2 < len(parts):
                        task_id = f"user_task_{parts[user_idx+2]}"

            except:
                pass
    
    elif "_none_none" in file and file.startswith("runs_"):
        is_utility = True
        # Format: runs_[DATASET_MODEL]_[AGENT]_user_task_[ID]_none_none...
        # Drift utility file
        target_agents = ['banking', 'travel', 'workspace', 'slack']
        found_agent_idx = -1
        
        for idx, part in enumerate(parts):
            if part in target_agents:
                found_agent_idx = idx
                agent_name = part
                break
        
        if found_agent_idx > 1:
            dataset_model = "_".join(parts[1:found_agent_idx])
            
            # extract task id: usually after agent
            # agent is at found_agent_idx. Next should be 'user', 'task', 'ID'
            if found_agent_idx + 3 < len(parts):
                if parts[found_agent_idx+1] == 'user' and parts[found_agent_idx+2] == 'task':
                    task_id = f"user_task_{parts[found_agent_idx+3]}"

    elif file.startswith("runs_"):
        is_utility = False
        # Format: runs_[DATASET_MODEL]_[AGENT]_...
        target_agents = ['banking', 'travel', 'workspace', 'slack']
        found_agent_idx = -1
        
        for idx, part in enumerate(parts):
            if part in target_agents:
                found_agent_idx = idx
                agent_name = part
                break
        
        if found_agent_idx > 1:
            dataset_model = "_".join(parts[1:found_agent_idx])
            
            # extract task id
            if found_agent_idx + 3 < len(parts):
                if parts[found_agent_idx+1] == 'user' and parts[found_agent_idx+2] == 'task':
                    task_id = f"user_task_{parts[found_agent_idx+3]}"
    
    if agent_name == "unknown":
         if "banking" in name_lower: agent_name = "banking"
         elif "travel" in name_lower: agent_name = "travel"
         elif "workspace" in name_lower: agent_name = "workspace"
         elif "slack" in name_lower: agent_name = "slack"

    return dataset_model, agent_name, is_utility, task_id

def calculate_correctness(data, is_utility):
    if not isinstance(data, list) or not data:
        return False
    
    max_val = 0
    for item in data:
        val = item.get("val", 0)
        if val > max_val:
            max_val = val
            
    if is_utility:
        return max_val == 0
    else:
        return max_val == 1

def update_stats(stats, system_name, dataset_model, agent_name, is_utility, is_correct):
    if dataset_model not in stats[system_name]:
        stats[system_name][dataset_model] = {}
    
    if agent_name not in stats[system_name][dataset_model]:
        stats[system_name][dataset_model][agent_name] = {
            'utility': {'correct': 0, 'total': 0},
            'defense': {'correct': 0, 'total': 0}
        }
    
    bucket = stats[system_name][dataset_model][agent_name]
    target = bucket['utility'] if is_utility else bucket['defense']
    
    target['total'] += 1
    if is_correct:
        target['correct'] += 1

def load_heuristic_data(input_dir, stats):
    heuristic_dir = os.path.join(input_dir, 'heuristic')
    
    # Map filename -> correctness (for direct matches)
    heuristic_direct_map = {}
    
    # Map (agent, task_id) -> list of {model: name, correct: bool} (for utility fuzzy matches)
    utility_lookup = defaultdict(list)
    
    if not os.path.exists(heuristic_dir):
        print(f"Warning: Heuristic directory not found at {heuristic_dir}")
        return heuristic_direct_map, utility_lookup
    
    print(f"Loading heuristic data from {heuristic_dir}...")
    for root, _, files in os.walk(heuristic_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
                
            try:
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    
                dataset_model, agent_name, is_utility, task_id = parse_filename_metadata(file)
                is_correct = calculate_correctness(data, is_utility)
                
                # STORE STATS FOR HEURISTIC (ALL FILES)
                update_stats(stats, 'heuristic', dataset_model, agent_name, is_utility, is_correct)
                
                # POPULATE LOOKUPS
                heuristic_direct_map[file] = is_correct
                
                if is_utility and agent_name != "unknown" and task_id != "unknown":
                    utility_lookup[(agent_name, task_id)].append({
                        'model': dataset_model,
                        'correct': is_correct,
                        'filename': file
                    })
                    
            except Exception as e:
                # print(f"Error reading heuristic file {file}: {e}")
                pass
                
    return heuristic_direct_map, utility_lookup

def process_drift_model_dir(drift_model_dir, drift_model, heuristic_data, stats):
    drift_system_key = drift_model
    heuristic_direct_map, utility_lookup = heuristic_data
    
    print(f"Scanning '{drift_model_dir}' for results (Drift Model: {drift_model})...")
    
    suffix = f"_{drift_model}.json"

    for root, _, files in os.walk(drift_model_dir):
        for file in files:
            if not file.endswith(suffix):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Identify type from current file
            dataset_model, agent_name, is_utility, task_id = parse_filename_metadata(file)
            drift_correct = calculate_correctness(data, is_utility)
            
            # STORE STATS FOR DRIFT MODEL (ALL FILES)
            update_stats(stats, drift_system_key, dataset_model, agent_name, is_utility, drift_correct)

            # --- MATCHING LOGIC FOR AGENT-SENTRY ---
            heuristic_correct = None
            
            if is_utility:
                # Fuzzy match for utility
                candidates = utility_lookup.get((agent_name, task_id), [])
                
                # Try Exact Match on dataset model first
                match = next((c for c in candidates if c['model'] == dataset_model), None)
                
                # Try StartsWith Match (Drift model is prefix of Heuristic model)
                if not match:
                     match = next((c for c in candidates if c['model'].startswith(dataset_model)), None)
                
                if match:
                    heuristic_correct = match['correct']
                    
            else:
                # Direct match for defense/other
                heuristic_filename = file[:-len(suffix)] + ".json"
                if heuristic_filename in heuristic_direct_map:
                    heuristic_correct = heuristic_direct_map[heuristic_filename]

            # Comput AGENT-SENTRY if match found
            if heuristic_correct is not None:
                sentry_correct = drift_correct or heuristic_correct
                update_stats(stats, 'agent-sentry', dataset_model, agent_name, is_utility, sentry_correct)

    return stats


def print_table(f, title, stats_data):
    f.write(f"{title}\n")
    f.write("="*140 + "\n")
    f.write(f"{'DATASET MODEL':<60} | {'AGENT':<15} | {'UTILITY ACCURACY':<25} | {'DEFENSE ACCURACY':<25}\n")
    f.write("-" * 140 + "\n")
    
    sorted_models = sorted(stats_data.keys())
    
    for d_model in sorted_models:
        agents = stats_data[d_model]
        for agent in sorted(agents.keys()):
            util_stats = agents[agent]['utility']
            def_stats = agents[agent]['defense']
            
            util_acc_str = "N/A"
            if util_stats['total'] > 0:
                acc = util_stats['correct'] / util_stats['total']
                util_acc_str = f"{acc:.2%} ({util_stats['correct']}/{util_stats['total']})"
            
            def_acc_str = "N/A"
            if def_stats['total'] > 0:
                acc = def_stats['correct'] / def_stats['total']
                def_acc_str = f"{acc:.2%} ({def_stats['correct']}/{def_stats['total']})"

            line = f"{d_model:<60} | {agent:<15} | {util_acc_str:<25} | {def_acc_str:<25}\n"
            f.write(line)
    f.write("="*140 + "\n\n")

def main():
    parser = argparse.ArgumentParser(description="Calculate drift statistics per agent.")
    parser.add_argument("--input_dir", type=str, default="drift_defense/evaluated_traces", help="Path to the evaluated traces directory.")
    parser.add_argument("--model", type=str, default=None, help="Optional: Specific drift model to process (folder name). If not set, processes all.")
    # ADDED ARGUMENT HERE
    parser.add_argument("--only_drift", action="store_true", help="If set, outputs only the drift statistics, excluding heuristic and agent-sentry.")
    
    args = parser.parse_args()

    if args.model:
        drift_models = [args.model]
    else:
        if not os.path.exists(args.input_dir):
             print(f"Error: Input directory '{args.input_dir}' does not exist.")
             return
        drift_models = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    if not drift_models:
        print("No drift model directories found.")
        return

    # Process Heuristic Data ONCE (Populates stats['heuristic'])
    # Shared stats dictionary
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'utility': {'total': 0, 'correct': 0},
        'defense': {'total': 0, 'correct': 0}
    })))
    
    heuristic_data_tuple = load_heuristic_data(args.input_dir, stats)

    # Process Drift Models
    for drift_model in drift_models:
        if drift_model == "heuristic": continue # Skip heuristic folder itself if listed
        
        target_dir = os.path.join(args.input_dir, drift_model)
        
        # Populate stats[drift_model] and stats['agent-sentry']
        process_drift_model_dir(target_dir, drift_model, heuristic_data_tuple, stats)

        # Report Generation per Drift Model
        output_stats_file = os.path.join(args.input_dir, drift_model, "drift_stats.txt")
        
        with open(output_stats_file, 'w') as f:
            # 1. Drift Model Stats (All) - Always printed
            print_table(f, f"Drift Statistics for Drift Model: {drift_model} (All Files)", stats[drift_model])
            f.write("\n")
            
            # CHECK FLAG BEFORE PRINTING OTHER TABLES
            if not args.only_drift:
                # 2. Heuristic Stats (All)
                print_table(f, f"Drift Statistics for Heuristic Model (All Files)", stats['heuristic'])
                f.write("\n")
                
                # 3. Agent-Sentry Stats (Matched Only)
                print_table(f, f"Drift Statistics for Agent-Sentry (Combined) (Matched Only)", stats['agent-sentry'])

        print(f"\nSaved stats for {drift_model} to {output_stats_file}")
        try:
            with open(output_stats_file, 'r') as f:
                print(f.read())
        except Exception as e:
            print(f"Error reading stats file: {e}")

if __name__ == "__main__":
    main()