#!/bin/bash

# --- CONFIGURATION ---
models=(
    #"gemma3:12b"
    #"gemma3:270m"
    #"gemma3:1b"
    #"gemma3:4b"
    #"smollm2:360m"
    #"llama3.2:3b"
    #"llama3.2:1b"
    #"mistral:7b"
    #"gemma3:27b"
    #"mistral-small3.2:24b"
    #"llama3.3:70b"
    #"smollm2:135m"
    #"ministral-3:3b"
    #"ministral-3:8b"
    #"gpt-5-mini"
    "gpt-5-nano"
)

# --- MAIN LOOP ---

# Ensure the log directory exists
mkdir -p drift_log_files

for model in "${models[@]}"; do
    safe_name=$(echo "$model" | tr ':' '_')
    log_file="drift_log_files/output_${safe_name}_agent_sentry_dataset.log"
    
    echo ">>> Launching (Fire & Forget): $model"

    # nohup ensures the process doesn't die when this script exits or you close the terminal.
    # The '&' at the end puts it in the background immediately.
    nohup ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        --model "$model" --agent_sentry_dataset \
        --inputs agent_sentry_dataset/ \
        --cot \
        --print > "$log_file" 2>&1 &

done

echo "---------------------------------------------------"
echo "All processes launched. Exiting script now."























        # ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        #     --model "$model" \
        #     --inputs agentdojo_simulated_data_full_utility/*.json \
        #     --print > "output_${safe_name}_ad_sim_util.log" 2>&1 &
        # pid_a=$!
        
        # ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        #     --model "$model" \
        #     --inputs agentdojo_simulated_data_full_attack/*extracted.json \
        #     --print > "output_${safe_name}_ad_sim_attack.log" 2>&1 &
        # pid_b=$!
        
        # ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        #     --model "$model" \
        #     --inputs utility_golden_traces_extracted_prompts_tools/*.json \
        #     --print > "output_${safe_name}_sim_util.log" 2>&1 &
        # pid_a=$!
        
        # ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        #     --model "$model" \
        #     --inputs attack_sim_combined_dataset_prompt_tools/*extracted.json \
        #     --print > "output_${safe_name}_sim_attack.log" 2>&1 &
        # pid_b=$!


        # ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        #     --model "$model" --agentdojo_format \
        #     --inputs generated_traces/runs/gpt-4o-2024-05-13/ \
        #     --security_true --utility_true --print > "output_${safe_name}_gpt-4o-2024-05-13_sim_attack.log" 2>&1 &
        # pid_a=$!

        # ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        #     --model "$model" --agentdojo_format \
        #     --inputs generated_traces/runs/gpt-4o-mini-2024-07-18/ \
        #     --security_true --utility_true --print > "output_${safe_name}_gpt-4o-mini-2024-07-18_sim_attack.log" 2>&1 &
        # pid_b=$!

        # ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        #     --model "$model" --agent_sentry_dataset \
        #     --inputs agent_sentry_dataset/ \
        #     --cot --print > "drift_log_files/output_${safe_name}_agent_sentry_dataset.log" 2>&1 &



        # ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        #     --model "$model" \
        #     --inputs golden_traces_extracted_prompt_tools_mimicry_attacks/*.json \
        #     --print > "output_${safe_name}_sim_mimicry.log" 2>&1 &
        # pid_c=$!

        # ~/myenv/bin/python3 drift_defense/run_drift_evaluation.py \
        #     --model "$model" \
        #     --inputs agentdojo_attacks_sim_datasets_prompts_tools/*extracted.json \
        #     --print > "output_${safe_name}_sim_attack.log" 2>&1 &
        # pid_a=$!

        #wait $pid_a $pid_b $pid_c

