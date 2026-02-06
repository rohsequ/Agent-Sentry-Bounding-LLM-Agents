print("DEBUG: Script starting...")
import argparse

print("DEBUG: Argparse imported")
import json
import os
import sys
import re
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Set

print("DEBUG: Stdlib imported")

# Add src to path to allow imports
src_path = Path(__file__).resolve().parent.parent / "src"
print(f"DEBUG: derived src path: {src_path}")
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
print("DEBUG: sys.path updated")

try:
    from utils.llm_client import LLMClient

    print("DEBUG: LLMClient imported")
    from pydantic import BaseModel

    print("DEBUG: Pydantic imported")
except Exception as e:
    print(f"DEBUG: Import error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Import the prompt dynamically
import importlib


def load_prompt_for_file(file_path: str, prompt_type: str = "default") -> str:
    """Loads the appropriate drift prompt based on the filename/path and generic type."""
    # prompt_type options: "default", "cot", "performance"
    # Check full path for agent name if not in filename (AgentDojo style)
    if "banking" in file_path.lower():
        module_name = "banking"
    elif "travel" in file_path.lower():
        module_name = "travel"
    elif "workspace" in file_path.lower():
        module_name = "workspace"
    elif "slack" in file_path.lower():
        module_name = "slack"
    else:
        # Fallback to filename check (legacy)
        filename = os.path.basename(file_path).lower()
        module_name = "email"  # Default
        if "banking" in filename:
            module_name = "banking"
        elif "travel" in filename:
            module_name = "travel"
        elif "workspace" in filename:
            module_name = "workspace"
        elif "slack" in filename:
            module_name = "slack"

    print(f"DEBUG: Selected prompt module '{module_name}' for file '{file_path}'")

    try:
        # Assuming drift_defense is in path as configured in main
        sys.path.append(str(Path(__file__).resolve().parent.parent))

        # Determine base package based on prompt_type
        if prompt_type == "cot":
            base_package = "drift_defense.cot_prompts"
        elif prompt_type == "performance":
            base_package = "drift_defense.performance_prompts"
        else:
            base_package = "drift_defense.prompts"

        module = importlib.import_module(f"{base_package}.{module_name}")
        return module.drift_prompt
    except Exception as e:
        print(f"DEBUG: Error loading prompt module {module_name}: {e}")
        sys.exit(1)


def get_agent_name(file_path: str) -> str:
    """Determines the agent name from the filename or path."""
    path_lower = file_path.lower()
    if "banking" in path_lower:
        return "banking"
    elif "travel" in path_lower:
        return "travel"
    elif "workspace" in path_lower:
        return "workspace"
    elif "slack" in path_lower:
        return "slack"
    return "email"


def get_high_risk_tools(agent_name: str) -> Set[str]:
    """
    Loads high risk tools for the specific agent.
    Tries to find drift_defense/tools/{agent_name}_tools.csv first.
    If not found or parse error, falls back to agentdojo_results_prompt_tools/high_risk_tools.csv.
    """
    repo_root = Path(__file__).resolve().parent.parent
    specific_tools_path = (
        repo_root / "drift_defense" / "tools" / f"{agent_name}_tools.csv"
    )
    fallback_tools_path = (
        repo_root / "agentdojo_results_prompt_tools" / "high_risk_tools.csv"
    )

    high_risk_tools = set()

    # Try agent-specific file first
    if specific_tools_path.exists():
        print(f"DEBUG: Loading high risk tools from {specific_tools_path}")
        try:
            with open(specific_tools_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # tool_name,risk,description (risk should be 'high')
                    if row.get("risk", "").strip().lower() == "high":
                        tool_name = row.get("tool_name", "").strip()
                        if tool_name:
                            high_risk_tools.add(tool_name)
            if high_risk_tools:
                return high_risk_tools
        except Exception as e:
            print(f"Warning: Error reading {specific_tools_path}: {e}")

    # Fallback
    if fallback_tools_path.exists():
        print(f"DEBUG: Loading high risk tools from fallback {fallback_tools_path}")
        try:
            with open(fallback_tools_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # id,high_risk_tool_name
                    tool_name = row.get("high_risk_tool_name", "").strip()
                    if tool_name:
                        high_risk_tools.add(tool_name)
        except Exception as e:
            print(f"Warning: Error reading {fallback_tools_path}: {e}")

    print(f"DEBUG: Loaded {len(high_risk_tools)} high risk tools")
    return high_risk_tools


class DriftScore(BaseModel):
    id: int
    val: float


class DriftResponse(BaseModel):
    scores: List[DriftScore]


from utils.config_loader import ConfigManager


def setup_llm_client(
    model_choice: str,
    use_local: bool = False,
    config_point: str = None,
    ollama_tag: str = None,
) -> LLMClient:
    """Configures and returns the LLMClient based on the model choice."""

    # Load config manager
    try:
        config_path = str(src_path / "configs" / "email_config.ini")
        config_manager = ConfigManager(config_path)
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        config_manager = None

    config = {}

    if use_local:
        if not config_point:
            # Default to localhost if not specified
            print(
                "Warning: --config_point not provided with --local. Defaulting to http://127.0.0.1:11434/v1"
            )
            config_point = "http://127.0.0.1:11434/v1"

        print(f"Using Local Override configuration with endpoint: {config_point}")

        # Map model choice to local model name if needed, or use as is
        local_model_name = model_choice
        if ollama_tag:
            local_model_name = ollama_tag
        elif model_choice == "llama3.3":
            local_model_name = "llama3.3:70b"
        elif model_choice == "gpt-oss-20b":
            local_model_name = "gpt-oss:20b"
        elif model_choice == "qwen-8b":
            local_model_name = "qwen3:8b"

        config = {
            "model_provider": "ollama",
            "model_name": local_model_name,
            "base_url": config_point,
            "api_key": "ollama",
            "temperature": 0.0,
            "max_tokens": 300,
        }

    elif model_choice == "llama3.3":
        # Strategy 1: Check DeepInfra Environment Variable
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if api_key:
            config = {
                "model_provider": "deepinfra",
                "model_name": "meta-llama/Llama-3.3-70B-Instruct",
                "base_url": "https://api.deepinfra.com/v1/openai",
                "api_key": api_key,
                "temperature": 0.0,
                "max_tokens": 300,
            }
        # Strategy 2: Check Config for Ollama
        elif (
            config_manager
            and config_manager.get("simple_agents", "model_provider") == "ollama"
        ):
            print("Using Ollama configuration from email_config.ini for Llama 3.3")
            config = {
                "model_provider": "ollama",
                "model_name": config_manager.get("simple_agents", "model_name"),
                "base_url": config_manager.get("simple_agents", "base_url"),
                "api_key": config_manager.get(
                    "simple_agents", "api_key", fallback="ollama"
                ),
                "temperature": 0.0,
                "max_tokens": 300,
            }
        else:
            raise ValueError(
                "API key for Llama 3.3 not found in env (DEEPINFRA_API_KEY) and no valid Ollama config found."
            )

    elif model_choice == "gpt-oss-20b":
        # Use Ollama configuration but override model name
        if (
            config_manager
            and config_manager.get("simple_agents", "model_provider") == "ollama"
        ):
            print(
                f"Using Ollama configuration from email_config.ini for {model_choice}"
            )
            config = {
                "model_provider": "ollama",
                "model_name": "gpt-oss:20b",  # Assuming this tag exists based on convention
                "base_url": config_manager.get("simple_agents", "base_url"),
                "api_key": config_manager.get(
                    "simple_agents", "api_key", fallback="ollama"
                ),
                "temperature": 0.0,
                "max_tokens": 300,
            }
        else:
            raise ValueError(
                "Ollama configuration not found in email_config.ini for gpt-oss-20b."
            )

    elif model_choice == "qwen-8b":
        # Use Ollama configuration
        if (
            config_manager
            and config_manager.get("simple_agents", "model_provider") == "ollama"
        ):
            print(
                f"Using Ollama configuration from email_config.ini for {model_choice}"
            )

            # Allow override or default to qwen3:8b
            tag = ollama_tag if ollama_tag else "qwen3:8b"

            config = {
                "model_provider": "ollama",
                "model_name": tag,
                "base_url": config_manager.get("simple_agents", "base_url"),
                "api_key": config_manager.get(
                    "simple_agents", "api_key", fallback="ollama"
                ),
                "temperature": 0.0,
                "max_tokens": 300,
            }
        else:
            raise ValueError(
                "Ollama configuration not found in email_config.ini for qwen-8b."
            )

    elif model_choice == "gpt-4o-mini":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and config_manager:
            # Fallback check in config (unlikely based on comments but possible)
            api_key = config_manager.get("models", "attack_eval_api_key", fallback=None)

        if api_key:
            config = {
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "api_key": api_key,
                "temperature": 0.0,
                "max_tokens": 300,
            }
        else:
            raise ValueError(
                "API key for gpt-4o-mini not found in environment variables (OPENAI_API_KEY)."
            )

    elif model_choice in ["gpt-5-nano", "gpt-5-mini"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key and config_manager:
            api_key = config_manager.get("models", "attack_eval_api_key", fallback=None)

        if api_key:
            config = {
                "model_provider": "openai",
                "model_name": model_choice,
                "api_key": api_key,
                "temperature": 0.0,
                "max_tokens": 300,
            }
        else:
            raise ValueError(
                f"API key for {model_choice} not found in environment variables (OPENAI_API_KEY)."
            )
    elif model_choice in [
        "mistral:7b",
        "mistral-small3.2:24b",
        "llama3.2:1b",
        "qwen2.5:0.5b",
        "qwen2.5:1.5b",
        "qwen2.5:3b",
        "llama3.2:3b",
        "smollm2:360m",
        "gemma3:12b",
        "gemma3:270m",
        "gemma3:1b",
        "gemma3:4b",
        "gemma3:27b",
        "llama3.3:70b",
        "smollm2:135m",
        "ministral-3:3b",
        "ministral-3:8b",
    ]:
        # Use Ollama configuration logic
        if (
            config_manager
            and config_manager.get("simple_agents", "model_provider") == "ollama"
        ):
            print(
                f"Using Ollama configuration from email_config.ini for {model_choice}"
            )
            config = {
                "model_provider": "ollama",
                "model_name": model_choice,
                "base_url": config_manager.get("simple_agents", "base_url"),
                "api_key": config_manager.get(
                    "simple_agents", "api_key", fallback="ollama"
                ),
                "temperature": 0.0,
                "max_tokens": 300,
            }
        else:
            raise ValueError(
                f"Ollama configuration not found in email_config.ini for {model_choice}."
            )

    else:
        raise ValueError(f"Unsupported model: {model_choice}")

    return LLMClient(config)


def main():
    print("DEBUG: main() started")
    parser = argparse.ArgumentParser(
        description="Run drift evaluation using specified LLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "llama3.3",
            "gpt-4o-mini",
            "gpt-oss-20b",
            "gpt-5-nano",
            "gpt-5-mini",
            "qwen-8b",
            "mistral:7b",
            "mistral-small3.2:24b",
            "llama3.2:1b",
            "qwen2.5:0.5b",
            "qwen2.5:1.5b",
            "qwen2.5:3b",
            "llama3.2:3b",
            "smollm2:360m",
            "gemma3:12b",
            "gemma3:270m",
            "gemma3:1b",
            "gemma3:4b",
            "gemma3:27b",
            "llama3.3:70b",
            "smollm2:135m",
            "ministral-3:3b",
            "ministral-3:8b",
        ],
        help="LLM model to use.",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="List of JSON files to parse.",
    )
    parser.add_argument(
        "--local", action="store_true", help="Use local LLM configuration."
    )
    parser.add_argument(
        "--config_point", type=str, help="API endpoint URL for local configuration."
    )
    parser.add_argument(
        "--ollama_tag",
        type=str,
        help="Specific Ollama model tag to use (overrides default mapping).",
    )
    parser.add_argument(
        "--print", action="store_true", help="Print the prompts sent to the LLM."
    )
    parser.add_argument(
        "--agentdojo_format",
        action="store_true",
        help="Parse AgentDojo trace files recursively from inputs.",
    )
    parser.add_argument(
        "--security_true",
        action="store_true",
        help="Only process files where security is True (AgentDojo specific).",
    )
    parser.add_argument(
        "--utility_true",
        action="store_true",
        help="Only process none.json files where utility is True (AgentDojo specific).",
    )
    parser.add_argument(
        "--agent_sentry_dataset",
        action="store_true",
        help="Enable loading for agent_sentry_dataset structure.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Explicitly set the dataset name for output directory naming (overrides inference from inputs).",
    )

    # Prompt variants
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use Chain-of-Thought prompts from drift_defense/cot_prompts.",
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Use Performance prompts from drift_defense/performance_prompts.",
    )

    args = parser.parse_args()

    if args.cot and args.performance:
        print("Error: Cannot specify both --cot and --performance.")
        return

    prompt_type = "default"
    if args.cot:
        prompt_type = "cot"
    elif args.performance:
        prompt_type = "performance"
    print(f"DEBUG: Args parsed: {args}")

    try:
        client = setup_llm_client(
            args.model, args.local, args.config_point, args.ollama_tag
        )
        print("DEBUG: LLMClient setup success")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    model_suffix = args.model.replace(".", "")  # llama3.3 -> llama33

    files_to_process = []
    if args.agentdojo_format:
        print("DEBUG: AgentDojo format enabled. expanding inputs...")
        for input_path in args.inputs:
            if os.path.isdir(input_path):
                # Walk directory to find .json files
                for root, dirs, files in os.walk(input_path):
                    for file in files:
                        if file.endswith(".json") and file != "stats.json":
                            # Allow none.json now (filtered in process_file)
                            files_to_process.append(os.path.join(root, file))
            elif os.path.isfile(input_path) and input_path.endswith(".json"):
                files_to_process.append(input_path)
            else:
                print(
                    f"Warning: Input path '{input_path}' not found or is not a file/directory."
                )
    elif args.agent_sentry_dataset:
        print("DEBUG: Agent Sentry Dataset mode enabled.")

        # Check if we were passed a list of files or a directory
        use_files_directly = False
        if args.inputs and len(args.inputs) > 0:
            if os.path.isfile(args.inputs[0]):
                use_files_directly = True

        if use_files_directly:
            print(f"DEBUG: Using {len(args.inputs)} provided files directly.")
            files_to_process = args.inputs
        else:
            # Determine root directory
            if args.inputs:
                root_dir = args.inputs[0]
            else:
                root_dir = "agent_sentry_dataset"

            if not os.path.exists(root_dir):
                print(f"Error: Directory '{root_dir}' not found.")
                return

            print(f"DEBUG: Walking directory {root_dir}...")
            for root, dirs, files in os.walk(root_dir):
                # strict filtering for attacks and utilities folders
                if "simulated_traces_full" in root:
                    continue

                path_parts = root.split(os.sep)
                if "attacks" in path_parts or "utilities" in path_parts:
                    for file in files:
                        if file.endswith(".json"):
                            files_to_process.append(os.path.join(root, file))
    else:
        files_to_process = args.inputs

    print(f"DEBUG: Processing {len(files_to_process)} files")
    for file_path in files_to_process:

        # Calculate output directory for AgentDojo format
        override_output_dir = None
        if args.agentdojo_format:
            # structure: evaluated_traces/<model_name>/<flattened_filename>
            # We pass the directory here
            override_output_dir = os.path.join("evaluated_traces", args.model)
        elif args.agent_sentry_dataset:
            # structure: [dataset_name]_drift_results/[model_name]/[attacks|utilities]/[agent_type]

            # Attempt to infer dataset name from the input path or default
            dataset_name = "agent_sentry_dataset"
            if args.dataset_name:
                dataset_name = args.dataset_name
            elif args.inputs:
                # Clean trailing slash if present
                clean_input = args.inputs[0].rstrip(os.sep)
                dataset_name = os.path.basename(clean_input)

            output_root = f"{dataset_name}_drift_results"

            # Parse path for category and agent
            parts = file_path.split(os.sep)
            category = "unknown"
            agent_type = "unknown"

            if "attacks" in parts:
                category = "attacks"
                idx = parts.index("attacks")
                if idx > 0:
                    agent_type = parts[idx - 1]
            elif "utilities" in parts:
                category = "utilities"
                idx = parts.index("utilities")
                if idx > 0:
                    agent_type = parts[idx - 1]

            override_output_dir = os.path.join(
                output_root, args.model, prompt_type, category, agent_type
            )

        process_file(
            client,
            file_path,
            args.model,
            model_suffix,
            drift_prompt=None,
            print_prompt=args.print,
            agentdojo_format=args.agentdojo_format,
            override_output_dir=override_output_dir,
            security_true=args.security_true,
            utility_true=args.utility_true,
            agent_sentry_dataset=args.agent_sentry_dataset,
            prompt_type=prompt_type,
        )


def extract_agentdojo_data(raw_data: dict, file_path: str) -> List[Dict[str, Any]]:
    """Extracts prompt and tools from AgentDojo structure."""
    prompt = ""
    tools_called = set()

    # Load high-risk tools to identify the action tool call
    agent_name = get_agent_name(file_path)
    high_risk_tools = get_high_risk_tools(agent_name)

    # Extract prompt from messages (System -> User (this is prompt) -> ...)
    messages = raw_data.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                prompt = content
            elif isinstance(content, list):
                # Handle list content (multimodal/etc)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        # AgentDojo uses "content" field, not "text"
                        prompt += part.get("content", part.get("text", ""))
            break

    # Extract tools from assistant messages
    # Keep track of tool call order to identify the action being tested
    tools_in_order = []
    for msg in messages:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = ""
                if isinstance(func, dict):
                    name = func.get("name", "")
                elif isinstance(func, str):
                    name = func

                if name:
                    tools_called.add(name)
                    tools_in_order.append(name)

    # Sort for consistency
    tools_str = ", ".join(sorted(list(tools_called)))

    # The action tool call is the first high-risk tool found in the sequence
    # (this is what triggers the intent alignment evaluation)
    action_tool_call = ""
    for tool in tools_in_order:
        if tool in high_risk_tools:
            action_tool_call = tool
            break

    # If no high-risk tools, just use empty string (will be marked as low-risk)

    # ID is not always present or numeric in these files, generate or use existing
    # AgentDojo files might not have an ID row per se, they are single trace usually
    # But process_file expects a list. We return a single item list.

    # Use absolute path or relative path from 'runs' for better traceability
    # User request: "runs/gpt-4o-mini-2024-07-18/travel/user_task_6/important_instructions/injection_task_5.json"

    relative_filename = file_path
    if "runs/" in file_path:
        # Slice from 'runs/' onwards
        relative_filename = file_path[file_path.find("runs/") :]
    elif "generated_traces/" in file_path:
        relative_filename = file_path[file_path.find("generated_traces/") :]

    # Debug: Print what we extracted
    print(f"  DEBUG: Extracted prompt length: {len(prompt)} chars")
    print(f"  DEBUG: Extracted tools: {tools_str}")
    print(f"  DEBUG: Action tool call: {action_tool_call}")
    print(f"  DEBUG: Filename: {relative_filename}")

    return [
        {
            "id": 1,
            "prompt": prompt,
            "tools": tools_str,
            "action_tool_call": action_tool_call,
            "filename": relative_filename,
        }
    ]


def extract_agent_sentry_data(raw_data: dict, file_path: str) -> List[Dict[str, Any]]:
    """Extracts prompt and tool calls from Agent Sentry dataset format."""
    prompt = ""
    tools_called = set()
    # Load high-risk tools to identify the action tool call
    agent_name = get_agent_name(file_path)
    high_risk_tools = get_high_risk_tools(agent_name)
    # 1. Inspect 'messages' (Attack files usually) OR 'conversation_history' (Utilities usually)
    messages = raw_data.get("messages", [])
    if not messages:
        messages = raw_data.get("conversation_history", [])

    # 2. Extract Prompt (First Human Message)
    for msg in messages:
        # Check type
        msg_type = msg.get("type")

        if msg_type == "human":
            # Extract content from data.content
            data = msg.get("data", {})
            content = data.get("content", "")
            if content:
                prompt = content
                break  # Found the prompt

    # 3. Extract Tool Calls (AI Messages)
    tools_list = []
    seen_tools = set()

    for msg in messages:
        msg_type = msg.get("type")

        if msg_type == "ai":
            data = msg.get("data", {})
            # Tool calls are usually in 'tool_calls' list
            tool_calls = data.get("tool_calls", [])
            for tc in tool_calls:
                name = tc.get("name", "")
                if name:
                    # Preserve order but deduplicate if multiple same calls exist?
                    # The user example implies distinct list. Safe to dedupe while preserving order.
                    if name not in seen_tools:
                        tools_list.append(name)
                        seen_tools.add(name)

    # Join in order
    tools_str = ", ".join(tools_list)

    # filename variable with the name of the file processed
    filename = os.path.basename(file_path)

    return [
        {
            "id": 1,  # Always use 1 for single-trace files in this dataset
            "prompt": prompt,
            "tools": tools_str,
            "filename": filename,
        }
    ]


def process_file(
    client: LLMClient,
    file_path: str,
    model_name: str,
    model_suffix: str,
    drift_prompt: str,
    print_prompt: bool = False,
    agentdojo_format: bool = False,
    override_output_dir: str = None,
    security_true: bool = False,
    utility_true: bool = False,
    agent_sentry_dataset: bool = False,
    prompt_type: str = "default",
):
    """Processes a single JSON file."""
    print(f"Processing {file_path}...")
    sys.stdout.flush()

    # If drift_prompt is None, load it dynamically (allows for different prompts per file if needed, which is the current design)
    if drift_prompt is None:
        drift_prompt = load_prompt_for_file(file_path, prompt_type)

    # ... rest of function ... we pass drift_prompt passed in or loaded here.
    # Actually checking main loop logic:
    # Original code:
    #   prompt_module = get_prompt_module(file_path) <-- wait, I didn't see this in my previous view of main loop, I saw load_prompt calls.
    #   Let's check lines 340-360 carefully.
    # Ah, I see I replaced the main loop iteration call.
    # BUT, I need to make sure I didn't break existing logic.
    # EXISTING MAIN Loop (inferred from previous view):
    #   process_file(..., drift_prompt, ...)
    # Wait, process_file takes drift_prompt.
    # In my replacement of main loop (args parsing), I didn't show the iteration part.
    # I need to verify if drift_prompt was calculated inside main loop or passed.

    # Let's look at lines 330-360 again using view_file to be safe before applying this chunk.
    # Skipping this chunk for now, will apply after verifying.
    pass
    """Processes a single JSON file."""
    print(f"Processing {file_path}...")
    sys.stdout.flush()

    # Load high risk tools for this agent type
    agent_name = get_agent_name(file_path)
    high_risk_tools = get_high_risk_tools(agent_name)

    try:
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        if agentdojo_format:
            filename = os.path.basename(file_path)

            # Logic for none.json
            if filename == "none.json":
                # Check 1: Must be in user_task folder
                if "user_task" not in file_path:
                    print(f"Skipping {file_path}: none.json not in user_task folder.")
                    return

                # Check 2: Utility flag (only applies to none.json)
                if utility_true:
                    is_utility = raw_data.get("utility", False)
                    if isinstance(is_utility, str):
                        is_utility = is_utility.lower() == "true"
                    elif not isinstance(is_utility, bool):
                        is_utility = False

                    if not is_utility:
                        print(
                            f"Skipping {file_path}: Utility is not true (requested --utility_true)."
                        )
                        return

            # Logic for other files (typically injection tasks)
            else:
                # Check Security filter (only applies to injection tasks)
                if security_true:
                    is_secure = raw_data.get("security", False)
                    # Check for "true" (string case-insensitive or boolean)
                    if isinstance(is_secure, str):
                        is_secure = is_secure.lower() == "true"
                    elif isinstance(is_secure, bool):
                        pass  # already boolean
                    else:
                        is_secure = False

                    if not is_secure:
                        print(
                            f"Skipping {file_path}: Security is not true (requested --security_true)."
                        )
                        return

            data = extract_agentdojo_data(raw_data, file_path)
        elif agent_sentry_dataset:
            data = extract_agent_sentry_data(raw_data, file_path)
        else:
            data = raw_data

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if not isinstance(data, list):
        print(
            f"Warning: {file_path} does not contain a list of interactions. Skipping."
        )
        return

    # Prepare output filename and directory
    if agentdojo_format:
        # Flat filename logic
        relative_path = file_path
        if "runs/" in file_path:
            relative_path = file_path[file_path.find("runs/") :]
        elif "generated_traces/" in file_path:
            relative_path = file_path[file_path.find("generated_traces/") :]

        # Replace slashes with underscores
        flat_name = relative_path.replace("/", "_")
        # Ensure it ends with json
        if not flat_name.endswith(".json"):
            flat_name += ".json"  # Should be there usually

        name_part, ext = os.path.splitext(flat_name)
        output_filename = f"{name_part}_{model_suffix}{ext}"
    else:
        base_name = os.path.basename(file_path)
        name_part, ext = os.path.splitext(base_name)
        output_filename = f"{name_part}_{model_suffix}{ext}"

    if override_output_dir:
        output_dir = override_output_dir
    else:
        # Create subfolder with model name
        output_dir = os.path.join(os.path.dirname(file_path), model_name)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)

    # Initialize output file with empty list if it doesn't exist
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            json.dump([], f)

    print(f"DEBUG: Starting loop for {len(data)} items")
    for i, item in enumerate(data):
        print(f"Processing row {i+1}/{len(data)}...")
        sys.stdout.flush()  # Ensure stdout is flushed (useful for nohup/logs)

        # Get ID from item, default to i+1 if missing
        item_id = item.get("id", i + 1)

        # Check for high risk tools
        tools_str = item.get("tools", "")
        if isinstance(tools_str, str):
            item_tools = [t.strip() for t in tools_str.split(",") if t.strip()]
        else:
            item_tools = []

        has_high_risk = any(tool in high_risk_tools for tool in item_tools)

        if not has_high_risk:
            print("  Skipping LLM call: No high-risk tools found. Assigning score 0.")
            parsed_scores = [
                {"val": 0, "response": "No LLM Call (Low Risk)", "inference_time": 0.0}
            ]
            if "filename" in item:
                parsed_scores[0]["filename"] = item["filename"]
            if "prompt" in item:
                parsed_scores[0]["prompt"] = item["prompt"]
            if "tools" in item:
                parsed_scores[0]["tools"] = item["tools"]
            if "action_tool_call" in item:
                parsed_scores[0]["action_tool_call"] = item["action_tool_call"]
            append_results(output_path, parsed_scores)
            continue

        # Prepare the single item prompt
        filtered_item = {
            "prompt": item.get("prompt", ""),
            "tools": item.get("tools", ""),
        }
        # REMOVE THE BRACKETS HERE
        single_item_dataset = json.dumps(filtered_item, indent=2)
        full_user_prompt = f"{drift_prompt}\n\n{single_item_dataset}"

        if print_prompt:
            print(f"--- Prompt for Row {i+1} ---")
            print(full_user_prompt)
            print("---------------------------")

        MAX_RETRIES = 3
        parsed_scores = None

        for attempt in range(MAX_RETRIES):
            try:
                # Accessing client.client directly:
                start_time = time.time()

                # Define the think setting based on the model
                extra_params = {}
                if "gpt-oss" in client.model_name:
                    extra_params["think"] = "low"  # Options: low, medium, high
                elif "qwen" in client.model_name:
                    extra_params["think"] = False  # Try to force-disable it for Qwen

                if "gpt-5" in client.model_name or "gpt-4o-mini" in client.model_name:
                    response = client.client.responses.create(
                        model=client.model_name,
                        input=[
                            {"role": "system", "content": drift_prompt},  # <--- Updated
                            {"role": "user", "content": single_item_dataset},
                        ],
                        extra_body={
                            "reasoning": {
                                "effort": "low"  # Options: "low", "medium", "high"
                            }
                        },
                    )
                    end_time = time.time()

                    # Debug logging for GPT-5 response structure
                    print(f"  DEBUG: Response type: {type(response)}")
                    # print(f"  DEBUG: Response repr: {repr(response)}") # Uncomment if needed, might be huge

                    # Extract content handling both legacy and new Response structures
                    content = ""
                    extracted = False

                    # Check for 'output' field (new structure)
                    if hasattr(response, "output"):
                        for output_item in response.output:
                            # Look for assistant messages
                            if (
                                getattr(output_item, "type", "") == "message"
                                and getattr(output_item, "role", "") == "assistant"
                            ):
                                # Iterate through content parts
                                for part in getattr(output_item, "content", []):
                                    if getattr(part, "type", "") == "output_text":
                                        content += getattr(part, "text", "")
                                        extracted = True

                    if not extracted:
                        if hasattr(response, "choices") and len(response.choices) > 0:
                            if hasattr(response.choices[0], "message"):
                                content = response.choices[0].message.content
                            elif hasattr(response.choices[0], "text"):
                                content = response.choices[0].text
                            else:
                                content = str(response.choices[0])
                        elif hasattr(response, "content"):
                            content = response.content
                        else:
                            content = str(response)

                    # Store raw answer for logging
                    raw_response_for_saving = content

                    # Clean explicit thinking tags
                    if content:
                        content = re.sub(
                            r"<think>.*?</think>", "", content, flags=re.DOTALL
                        )
                        content = re.sub(
                            r"^Here is my thought process:.*",
                            "",
                            content,
                            flags=re.DOTALL,
                        )

                    raw_content = content  # For consistency

                else:
                    response = client.client.chat.completions.create(
                        model=client.model_name,
                        messages=[
                            {"role": "system", "content": drift_prompt},  # <--- Updated
                            {"role": "user", "content": single_item_dataset},
                        ],
                        temperature=client.config["temperature"],
                        max_tokens=client.config["max_tokens"],
                        extra_body=extra_params,  # <--- ADD THIS
                    )
                    end_time = time.time()
                    content = response.choices[0].message.content
                    # --- CHANGE 2: Strip <think> tags before processing ---
                    # Even with the prompt, reasoning models often drift into thinking.
                    # We strip this block out so we don't parse numbers from the thought process.
                    raw_response_for_saving = (
                        content  # Save original content before stripping
                    )
                    if content:
                        # Remove content between <think> and </think> (including the tags)
                        content = re.sub(
                            r"<think>.*?</think>", "", content, flags=re.DOTALL
                        )

                        # Also remove generic thinking preambles if the model doesn't use tags
                        content = re.sub(
                            r"^Here is my thought process:.*",
                            "",
                            content,
                            flags=re.DOTALL,
                        )
                    raw_content = content
                    content = content.strip() if content else ""

                inference_time = end_time - start_time
                print(f"  Inference time: {inference_time:.2f}s")
                content = content.strip()

                # Clean content markdown
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                if not content:
                    print(
                        f"  Attempt {attempt+1}/{MAX_RETRIES}: Received empty response (raw_len={len(raw_content)}). Raw content: {repr(raw_content)}. Retrying..."
                    )
                    continue

                # Try to parse as single digit
                digit_match = re.search(r"\b([0-9])\b", content)

                if digit_match:
                    score_val = int(digit_match.group(1))
                    parsed_scores = [
                        {
                            "id": item_id,
                            "val": score_val,
                            "response": raw_response_for_saving,
                            "inference_time": inference_time,
                        }
                    ]
                    break

                # Fallback: try parsing as JSON
                try:
                    json_content = json.loads(content)
                    if isinstance(json_content, list):
                        for res in json_content:
                            if isinstance(res, dict):
                                res["response"] = raw_response_for_saving
                                res["inference_time"] = inference_time
                        parsed_scores = json_content
                        break
                    elif isinstance(json_content, (int, float)):
                        parsed_scores = [
                            {
                                "id": item_id,
                                "val": json_content,
                                "response": raw_response_for_saving,
                                "inference_time": inference_time,
                            }
                        ]
                        break
                    elif isinstance(json_content, dict):
                        # If dictionary, looks for val/score, otherwise skip
                        val = json_content.get("val", json_content.get("score"))
                        if val is not None:
                            parsed_scores = [
                                {
                                    "id": item_id,
                                    "val": val,
                                    "response": raw_response_for_saving,
                                    "inference_time": inference_time,
                                }
                            ]
                            break
                except json.JSONDecodeError as e:
                    pass

                print(
                    f"  Attempt {attempt+1}/{MAX_RETRIES}: Could not parse score from: {content}"
                )
                if attempt == MAX_RETRIES - 1:
                    parsed_scores = None
                continue

            except Exception as e:
                print(
                    f"  Attempt {attempt+1}/{MAX_RETRIES}: Error calling LLM ({client.model_name} at {client.config.get('base_url')}): {e}"
                )
                if attempt == MAX_RETRIES - 1:
                    parsed_scores = None
                continue

        if parsed_scores is None:
            print(f"  Failed to process item {i} after {MAX_RETRIES} attempts.")
            continue

        # Add filename, prompt, and tools to results if available
        if parsed_scores:
            print(
                f"  DEBUG: Item type: {type(item)}, has 'prompt': {'prompt' in dir(item) if not isinstance(item, dict) else 'prompt' in item}"
            )
            for score in parsed_scores:
                # Remove ID if present (user requested removal)
                if "id" in score:
                    del score["id"]

                # Always add prompt and tools from the item
                if "prompt" in item:
                    score["prompt"] = item["prompt"]
                    print(f"  DEBUG: Added prompt field ({len(item['prompt'])} chars)")
                if "tools" in item:
                    score["tools"] = item["tools"]
                    print(f"  DEBUG: Added tools field: {item['tools']}")
                if "action_tool_call" in item:
                    score["action_tool_call"] = item["action_tool_call"]
                    print(
                        f"  DEBUG: Added action_tool_call field: {item['action_tool_call']}"
                    )
                if "filename" in item:
                    score["filename"] = item["filename"]
                    print(f"  DEBUG: Added filename field: {item['filename']}")

        # Append results incrementally
        if isinstance(parsed_scores, list):
            append_results(output_path, parsed_scores)
        else:
            print(f"  Warning: Unexpected output format: {type(parsed_scores)}")


def append_results(file_path: str, new_results: List[Any]):
    """Appends new results to a JSON list file efficiently."""
    try:
        # Read existing data
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.extend(new_results)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"Error appending to {file_path}: {e}")


if __name__ == "__main__":
    main()
