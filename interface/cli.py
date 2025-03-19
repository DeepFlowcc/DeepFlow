"""
Command Line Interface Module
===========================

This module provides a command-line interface for running agents with different models
and configurations. It enables users to interact with the agent framework directly 
from the terminal, specifying parameters like model type, model ID, authorized imports,
and tools the agent can use.

The CLI is designed to be flexible and user-friendly, allowing quick experimentation
with different agent configurations without writing code.

DeepFlow@2025
"""

#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard library imports
import argparse  # For command-line argument parsing
import os  # For environment variable access and file path handling

# Third-party imports
from dotenv import load_dotenv  # For loading environment variables from .env files

# Local imports from the agent framework
from smolagents import CodeAgent, HfApiModel, LiteLLMModel, Model, OpenAIServerModel, Tool, TransformersModel
from smolagents.default_tools import TOOL_MAPPING  # Dictionary of available built-in tools


# Default prompt used when no prompt is provided by the user
DEFAULT_PROMPT = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"


def parse_arguments():
    """
    Parse command line arguments.
    
    This function defines all the command-line arguments available for configuring
    the agent, including model type, model ID, authorized imports, tools, and API options.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    # Create a new argument parser with a description
    parser = argparse.ArgumentParser(description="Run a CodeAgent with all specified parameters")
    
    # Main arguments
    # The prompt argument is positional and optional (with a default value)
    parser.add_argument(
        "prompt",  # The name of the argument
        type=str,  # The argument should be interpreted as a string
        nargs="?",  # Makes the argument optional
        default=DEFAULT_PROMPT,  # Default value if not provided
        help="The prompt to run with the agent",  # Help text for the argument
    )
    
    # The model type argument (e.g., HfApiModel, OpenAIServerModel)
    parser.add_argument(
        "--model-type",  # The flag for this argument
        type=str,  # The argument should be interpreted as a string
        default="HfApiModel",  # Default value if not provided
        help="The model type to use (e.g., HfApiModel, OpenAIServerModel, LiteLLMModel, TransformersModel)",
    )
    
    # The model ID argument (e.g., specific model name/path)
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        help="The model ID to use for the specified model type",
    )
    
    # Additional Python packages to authorize for import in the agent
    parser.add_argument(
        "--imports",
        nargs="*",  # Accept zero or more values
        default=[],  # Default to an empty list
        help="Space-separated list of imports to authorize (e.g., 'numpy pandas')",
    )
    
    # Tools that the agent can use
    parser.add_argument(
        "--tools",
        nargs="*",  # Accept zero or more values
        default=["web_search"],  # Default to web_search tool
        help="Space-separated list of tools that the agent can use (e.g., 'tool1 tool2 tool3')",
    )
    
    # Verbosity level for the agent's output
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="The verbosity level, as an int in [0, 1, 2].",
    )
    
    # API-specific options in a separate group for better organization
    api_group = parser.add_argument_group("api options", "Options for API-based model types")
    
    # Base URL for the API
    api_group.add_argument(
        "--api-base",
        type=str,
        help="The base URL for the model",
    )
    
    # API key for authentication
    api_group.add_argument(
        "--api-key",
        type=str,
        help="The API key for the model",
    )
    
    # Parse the arguments from the command line and return them
    return parser.parse_args()


def create_model(model_type: str, model_id: str, api_base: str | None = None, api_key: str | None = None) -> Model:
    """
    Create a model instance based on model type and parameters.
    
    This function instantiates the appropriate model class based on the specified
    model type and configures it with the provided parameters.
    
    Args:
        model_type: Type of model to create (e.g., "OpenAIServerModel", "HfApiModel")
        model_id: ID of the model to use
        api_base: Optional base URL for API-based models
        api_key: Optional API key for API-based models
        
    Returns:
        Model: Instantiated model
        
    Raises:
        ValueError: If an unsupported model type is specified
    """
    # Handle OpenAI-compatible APIs (like Fireworks)
    if model_type == "OpenAIServerModel":
        return OpenAIServerModel(
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),  # Use provided key or get from environment
            api_base=api_base or "https://api.fireworks.ai/inference/v1",  # Use provided base or default
            model_id=model_id,
        )
    # Handle LiteLLM models (a library that provides access to various LLM APIs)
    elif model_type == "LiteLLMModel":
        return LiteLLMModel(
            model_id=model_id,
            api_key=api_key,
            api_base=api_base,
        )
    # Handle locally-run Transformers models
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto")  # Auto device placement
    # Handle Hugging Face API models
    elif model_type == "HfApiModel":
        return HfApiModel(
            model_id=model_id,
            token=api_key or os.getenv("HF_API_KEY"),  # Use provided key or get from environment
        )
    # Raise an error for unsupported model types
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main(
    prompt: str,
    tools: list[str],
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    imports: list[str] | None = None,
) -> None:
    """
    Main function to run the agent with specified parameters.
    
    This function sets up the model, loads the specified tools, creates the agent,
    and runs it with the provided prompt.
    
    Args:
        prompt: The prompt to send to the agent
        tools: List of tool names to enable for the agent
        model_type: Type of model to use
        model_id: ID of the model to use
        api_base: Optional base URL for API-based models
        api_key: Optional API key for API-based models
        imports: Optional list of Python modules the agent is allowed to import
    """
    # Load environment variables from .env file if present
    load_dotenv()

    # Create the model based on specified parameters
    model = create_model(model_type, model_id, api_base=api_base, api_key=api_key)

    # Load and initialize the specified tools
    agent_tools = []
    for tool_name in tools:
        if "/" in tool_name:
            # Tool is a HuggingFace Space (identified by slash in the name)
            agent_tools.append(Tool.from_space(tool_name))
        else:
            # Tool is a default tool from TOOL_MAPPING
            if tool_name in TOOL_MAPPING:
                agent_tools.append(TOOL_MAPPING[tool_name]())  # Instantiate the tool
            else:
                raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

    # Display tools being used for user feedback
    print(f"Running agent with these tools: {tools}")
    
    # Create and run the agent with the specified configuration
    agent = CodeAgent(tools=agent_tools, model=model, additional_authorized_imports=imports)
    agent.run(prompt)  # Execute the agent with the given prompt


# Entry point for command-line execution
if __name__ == "__main__":
    # Parse command line arguments and run main function
    args = parse_arguments()

    # Call the main function with the parsed arguments
    main(
        args.prompt,
        args.tools,
        args.model_type,
        args.model_id,
        api_base=args.api_base,
        api_key=args.api_key,
        imports=args.imports,
    )
