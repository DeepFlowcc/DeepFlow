#!/usr/bin/env python
# coding=utf-8

"""
DeepFlow Core Package
====================

This package provides a comprehensive framework for building intelligent agents that can 
perform complex tasks through a combination of planning, reasoning, and tool usage.

The package structure follows a modular design where each component handles specific 
functionality within the agent ecosystem:

- agent_types: Defines data types that can be returned by agents
- agents: Implements the core agent classes (MultiStepAgent, ToolCallingAgent, CodeAgent)
- tools: Provides the base Tool class and utility functions for tool creation
- models: Contains model classes for different LLM providers
- memory: Implements memory structures for agent state management
- monitoring: Provides logging and monitoring capabilities

Version: 1.12.0.dev0

DeepFlow@2025
"""

# Define the package version
__version__ = "1.12.0.dev0"

# Import all modules to make them available via the top-level package
# This allows users to import directly from the package, e.g. from smolagents import MultiStepAgent
from .agent_types import *  # Import agent data type classes (AgentImage, AgentText, etc.)
from .agents import *       # Import agent implementation classes (MultiStepAgent, ToolCallingAgent, CodeAgent)
from .default_tools import * # Import common pre-built tools (PythonInterpreter, WebSearch, etc.)
from .gradio_ui import *     # Import UI components for Gradio integration
from .local_python_executor import * # Import execution environment for Python code
from .memory import *        # Import memory-related classes for state management
from .models import *        # Import model interfaces for different providers
from .monitoring import *    # Import logging and monitoring utilities
from .remote_executors import * # Import remote execution environments
from .tools import *         # Import base tool classes and tool utilities
from .utils import *         # Import general utility functions
from .cli import *           # Import command-line interface utilities
