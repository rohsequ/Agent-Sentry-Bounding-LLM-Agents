# Agent Sentry - Research Artifact

This repository contains the code and datasets for the Agent Sentry research project, which evaluates security and utility tradeoffs in agentic systems using functionaltiy graph defenses with intent alignment.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Core Scripts](#core-scripts)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Results Locations](#results-locations)

## Overview

This artifact includes:
- **Agent Sentry Dataset**: Custom dataset of agent traces with utilities and attacks
- **AgentDojo Runs**: Three independent runs (AD_run1, AD_run2, AD_run3) of the AgentDojo benchmark
- **Intent Alignment Module**: LLM-based intent drift detection
- **Functionality Graph Learning**: Security-utility tradeoff experiments

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

Install all required packages using the provided `requirements.txt` file:

```bash
# Navigate to the repository
cd /path/to/agent_sentry_submission

# Install all dependencies
pip install -r requirements.txt
```

**Required Packages** (automatically installed from `requirements.txt`):
- **numpy** (>=1.21.0): Numerical computations and array operations
- **matplotlib** (>=3.5.0): Plotting and visualization for security-utility tradeoff curves
- **pandas** (>=1.3.0): Data manipulation and analysis for CSV outputs
- **openai** (>=1.0.0): OpenAI API client for GPT models used in intent alignment
- **langchain-core** (>=0.1.0): Core LangChain abstractions for message handling
- **langchain** (>=0.1.0): LangChain framework for building LLM applications
- **langgraph** (>=0.0.1): Graph-based agent orchestration
- **pydantic** (>=2.0.0): Data validation and settings management
- **rich** (>=13.0.0): Beautiful terminal formatting and output
- **requests** (>=2.28.0): HTTP library for API calls

**Optional Dependencies**:
- **agentdojo**: AgentDojo benchmark framework (uncomment in `requirements.txt` when needed)

**Note**: You will need to set up your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Directory Structure

The repository is organized as follows:

```
agent_sentry_submission/
├── agent_sentry_clean_unique_dataset/    # Agent Sentry dataset (traces)
│   ├── banking/
│   ├── slack/
│   ├── travel/
│   └── workspace/
├── AD_run1/                              # AgentDojo run 1 traces
├── AD_run2/                              # AgentDojo run 2 traces
├── AD_run3/                              # AgentDojo run 3 traces
├── intent_alignment/                      # Intent alignment module
├── intent_alignment_unique_results/       # Intent results for Agent Sentry
├── agentdojo_intent_alignment_results/   # Intent results for AgentDojo
├── deterministic_policy/                  # Functionality graph learning and evaluation
│   ├── experiments/
│   │   └── balanced_experiments_50_100/  # Agent Sentry paper results
│   ├── AD1_tradeoff_results_latest/      # AgentDojo Run 1 paper results
│   ├── AD2_tradeoff_results_latest/      # AgentDojo Run 2 paper results
│   └── AD3_tradeoff_results_latest/      # AgentDojo Run 3 paper results
└── src/                                   # Core utilities and functionality graphs
```

## Datasets

### Agent Sentry Dataset

**Location**: `agent_sentry_clean_unique_dataset/`

Contains agent execution traces organized by:
- **Agents**: banking, slack, travel, workspace
- **Categories**: utilities (benign tasks) and attacks (malicious tasks)

Each trace is a JSON file containing:
- Tool calls and parameters
- User instructions
- Agent responses
- Ground truth labels

### AgentDojo Runs

**Locations**: `AD_run1/`, `AD_run2/`, `AD_run3/`

Three independent runs of the AgentDojo benchmark containing:
- User task directories (`user_task_*`)
- Utility traces (`none/none.json` with `utility=True`)
- Attack traces (`important_instructions/*.json` with `security=True`)

## Core Scripts

### 1. Intent Alignment for Agent Sentry Dataset

**Script**: `run_intent_alignment_full.py`

**Purpose**: Generate intent alignment results for the Agent Sentry dataset by detecting intent drift using an LLM.

**Usage**:
```bash
python run_intent_alignment_full.py
```

**What it does**:
1. Scans `agent_sentry_clean_unique_dataset/` for all trace files
2. Compares against existing results in `intent_alignment_unique_results/`
3. Identifies missing files that haven't been processed
4. Runs intent alignment analysis on missing files using the drift detection module
5. Saves results to `intent_alignment_unique_missing/`

**Configuration**:
- Model: `gpt-5-mini` (configurable in script)
- Batch size: 500 files per batch
- Mode: Chain-of-thought (CoT) prompting

**Output**: Intent alignment results in `intent_alignment_unique_results/` with format:
- Input: `agent_sentry_clean_unique_dataset/{agent}/{category}/file.json`
- Output: `intent_alignment_unique_results/{agent}/{category}/file_gpt-5-mini.json`

---

### 2. Intent Alignment for AgentDojo Dataset

**Script**: `run_agentdojo_intent_alignment.py`

**Purpose**: Generate intent alignment results for all three AgentDojo runs.

**Usage**:
```bash
python run_agentdojo_intent_alignment.py \
    --model gpt-5-nano \
    --mode cot \
    --ad_dirs AD_run1 AD_run2 AD_run3 \
    --output_root agentdojo_intent_alignment_results
```

**Parameters**:
- `--model`: Model to use for intent alignment (default: `gpt-5-nano`)
  - Options: `gpt-5-nano`, `gpt-5-mini`, `gpt-4o-mini`, `llama3.3`, `mistral:7b`, `gemma3:12b`
- `--mode`: Prompt mode (default: `cot`)
  - Options: `default`, `cot`, `performance`
- `--ad_dirs`: AgentDojo directories to process (default: all three runs)
- `--output_root`: Output directory (default: `agentdojo_intent_alignment_results`)

**What it does**:
1. Processes each AgentDojo run (AD_run1, AD_run2, AD_run3)
2. Identifies all utility and attack traces
3. Runs intent alignment analysis using the specified model
4. Saves results in a structured format compatible with the evaluation scripts

**Output**: Results saved to `agentdojo_intent_alignment_results/{AD_run}/evaluated_traces/{model}/`

---

### 3. Learn Functionality Graph on Agent Sentry Dataset

**Script**: `deterministic_policy/train_balanced_tradeoff.py`

**Purpose**: Learn and evaluate functionality graphs on the Agent Sentry dataset with varying security-utility tradeoffs.

**Usage**:
```bash
cd deterministic_policy

python train_balanced_tradeoff.py \
    --dataset_root ../agent_sentry_clean_unique_dataset \
    --intent_root ../intent_alignment_unique_results \
    --output_dir experiments/balanced_experiments_50_100 \
    --mode heuristic \
    --workers 100 \
    --seed 42
```

**Parameters**:
- `--dataset_root`: Path to Agent Sentry dataset
- `--intent_root`: Path to intent alignment results
- `--output_dir`: Directory for experiment outputs
- `--mode`: Functionality graph mode (`heuristic`, `threshold`, or `hybrid`)
- `--workers`: Number of parallel workers for extraction
- `--seed`: Random seed for reproducibility (paper uses seeds: 42, 43, 44)

**What it does**:
1. Loads traces from the Agent Sentry dataset
2. Incorporates intent alignment results to improve classification
3. Learns functionality graphs with different utility fractions (10%-100%)
4. Evaluates security (attack blocking) and utility (benign task success)
5. Generates comprehensive results including:
   - Per-agent performance metrics
   - Security-utility tradeoff curves
   - Statistical summaries
   - Visualization plots

**Output**: Results saved to `deterministic_policy/experiments/balanced_experiments_50_100/`

**Key Output Files**:
- `aggregated_experiment_summary.json`: Overall statistics across all experiments
- `detailed_summary_results.csv`: Per-configuration metrics
- `{agent}/fraction_{frac}/seed_{seed}/`: Individual experiment results
  - `policy.json`: Learned functionality graph parameters
  - `metrics.json`: Performance metrics
  - `evaluation_results.json`: Detailed evaluation data

---

### 4. Learn Functionality Graph on AgentDojo Dataset

**Script**: `deterministic_policy/train_agentdojo_security_utility_tradeoff_with_intent.py`

**Purpose**: Learn and evaluate functionality graphs on the AgentDojo benchmark runs.

**Usage**:

For each AgentDojo run, execute:

```bash
cd deterministic_policy

# Run 1
python train_agentdojo_security_utility_tradeoff_with_intent.py \
    --dataset_root ../AD_run1/gpt-4o-2024-05-13 \
    --intent_root ../agentdojo_intent_alignment_results/AD_run1 \
    --output_dir AD1_tradeoff_results_latest \
    --mode heuristic \
    --workers 100 \
    --seed 42

# Run 2
python train_agentdojo_security_utility_tradeoff_with_intent.py \
    --dataset_root ../AD_run2/gpt-4o-2024-05-13 \
    --intent_root ../agentdojo_intent_alignment_results/AD_run2 \
    --output_dir AD2_tradeoff_results_latest \
    --mode heuristic \
    --workers 100 \
    --seed 42

# Run 3
python train_agentdojo_security_utility_tradeoff_with_intent.py \
    --dataset_root ../AD_run3/gpt-4o-2024-05-13 \
    --intent_root ../agentdojo_intent_alignment_results/AD_run3 \
    --output_dir AD3_tradeoff_results_latest \
    --mode heuristic \
    --workers 100 \
    --seed 42
```

**Parameters**:
- `--dataset_root`: Path to AgentDojo run directory (with model subdirectory)
- `--intent_root`: Path to intent alignment results for this run
- `--output_dir`: Directory for experiment outputs
- `--mode`: Functionality graph mode (`heuristic`, `threshold`, or `hybrid`)
- `--workers`: Number of parallel workers for extraction
- `--seed`: Random seed for reproducibility (paper uses seeds: 42, 43, 44)

**What it does**:
1. Loads AgentDojo traces (utility and attack)
2. Incorporates intent alignment results
3. Learns functionality graphs with varying utility learning fractions (10%-100%)
4. Evaluates on full utility and attack test sets
5. Generates comprehensive security-utility tradeoff analysis

**Output**: Results saved to:
- `deterministic_policy/AD1_tradeoff_results_latest/`
- `deterministic_policy/AD2_tradeoff_results_latest/`
- `deterministic_policy/AD3_tradeoff_results_latest/`

## Reproducing Paper Results

### Step-by-Step Guide

#### 1. Generate Intent Alignment Results

**For Agent Sentry Dataset**:
```bash
python run_intent_alignment_full.py
```

**For AgentDojo Datasets**:
```bash
python run_agentdojo_intent_alignment.py \
    --model gpt-5-nano \
    --mode cot
```

⏱️ **Time estimate**: Several hours depending on dataset size and API rate limits

#### 2. Learn Functionality Graphs and Generate Results

**For Agent Sentry Dataset**:
```bash
cd deterministic_policy

python train_balanced_tradeoff.py \
    --dataset_root ../agent_sentry_clean_unique_dataset \
    --intent_root ../intent_alignment_unique_results \
    --output_dir experiments/balanced_experiments_50_100 \
    --mode heuristic \
    --workers 100 \
    --seed 42
```

⏱️ **Time estimate**: 1-3 hours

**For AgentDojo Datasets** (all three runs):
```bash
cd deterministic_policy

# Process each run sequentially
for run in 1 2 3; do
    python train_agentdojo_security_utility_tradeoff_with_intent.py \
        --dataset_root ../AD_run${run}/gpt-4o-2024-05-13 \
        --intent_root ../agentdojo_intent_alignment_results/AD_run${run} \
        --output_dir AD${run}_tradeoff_results_latest \
        --mode heuristic \
        --workers 100 \
        --seed 42
done
```

⏱️ **Time estimate**: 2-4 hours per run

#### 3. Analyze Results

Results are automatically saved with comprehensive metrics, plots, and summaries. Key files to examine:

- `aggregated_experiment_summary.json`: High-level statistics
- `detailed_summary_results.csv`: Per-configuration metrics (easy to import into spreadsheet tools)
- Individual experiment directories contain detailed breakdowns

## Results Locations

### Paper Results - Agent Sentry Dataset

**Location**: `deterministic_policy/experiments/balanced_experiments_50_100/`

Contains:
- Security-utility tradeoff results for all agents (banking, slack, travel, workspace)
- Multiple learning fractions (10% to 100%)
- Multiple random seeds for statistical significance: **42, 43, 44**
- Aggregated summaries and visualizations

### Paper Results - AgentDojo Benchmark

**Location**: Three separate directories for each run:

1. **Run 1**: `deterministic_policy/AD1_tradeoff_results_latest/`
2. **Run 2**: `deterministic_policy/AD2_tradeoff_results_latest/`
3. **Run 3**: `deterministic_policy/AD3_tradeoff_results_latest/`

Each contains:
- Security-utility tradeoff results for all AgentDojo agents
- Multiple learning configurations
- Statistical summaries across all experiments

### Intent Alignment Results

**Agent Sentry**: `intent_alignment_unique_results/`
- Intent drift detection results for Agent Sentry dataset
- Format: `{agent}/{category}/{filename}_gpt-5-mini.json`

**AgentDojo**: `agentdojo_intent_alignment_results/`
- Intent drift detection results for all three AgentDojo runs
- Format: `AD_run{1,2,3}/evaluated_traces/{model}/{flattened_filename}.json`

## Additional Information

### Understanding the Results

Each experiment directory contains:

1. **policy.json**: Learned functionality graph parameters including:
   - Feature frequencies
   - Penalty weights
   - Threshold values

2. **metrics.json**: Performance metrics including:
   - Security rate (% attacks blocked)
   - Utility rate (% benign tasks successful)
   - F1 scores
   - Confusion matrix values

3. **evaluation_results.json**: Detailed per-trace evaluation data

### Reproducibility

For reproducibility, all experiments in the paper were run with the following seed values:
- **Primary seed**: 42 (used in all example commands)
- **Additional seeds for statistical significance**: 43, 44

To reproduce exact paper results, run experiments with all three seeds:
```bash
for seed in 42 43 44; do
    python train_balanced_tradeoff.py \
        --dataset_root ../agent_sentry_clean_unique_dataset \
        --intent_root ../intent_alignment_unique_results \
        --output_dir experiments/balanced_experiments_50_100 \
        --mode heuristic \
        --workers 100 \
        --seed $seed
done
```

The aggregated results combine statistics across all seed runs to provide robust performance estimates with confidence intervals.
