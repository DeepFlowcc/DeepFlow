#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Monitoring and Logging Module
===========================

This module provides comprehensive logging and monitoring capabilities for agents.
It includes classes for tracking metrics, logging agent activities at different
verbosity levels, and visualizing agent structures.

The monitoring system helps track performance metrics like token usage and execution times,
while the logging system provides formatted output for debugging and user feedback.

DeepFlow@2025
"""

import json
from enum import IntEnum
from typing import List, Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from smolagents.utils import escape_code_brackets


__all__ = ["AgentLogger", "LogLevel", "Monitor"]


class Monitor:
    """
    Tracks and records metrics for agent performance.
    
    This class monitors various performance metrics like execution times,
    token counts, and provides methods to query accumulated statistics.
    """
    
    def __init__(self, tracked_model, logger):
        """
        Initialize a monitor instance.
        
        Args:
            tracked_model: The model to track metrics for
            logger: The logger to output monitoring information
        """
        self.step_durations = []
        self.tracked_model = tracked_model
        self.logger = logger
        if hasattr(self.tracked_model, "last_input_token_count"):
            self.total_input_token_count = 0
            self.total_output_token_count = 0

    def get_total_token_counts(self):
        """
        Get the accumulated token counts for both input and output.
        
        Returns:
            dict: A dictionary with 'input' and 'output' token counts
        """
        return {
            "input": self.total_input_token_count,
            "output": self.total_output_token_count,
        }

    def reset(self):
        """
        Reset all accumulated metrics to their initial state.
        """
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def update_metrics(self, step_log):
        """
        Update metrics based on the latest step execution.
        
        This method records execution time and token counts from a step,
        updating the accumulated statistics.
        
        Args:
            step_log: A MemoryStep containing execution data
        """
        step_duration = step_log.duration
        self.step_durations.append(step_duration)
        metrics_message = f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"

        if hasattr(self.tracked_model, "last_input_token_count"):
            self.total_input_token_count += self.tracked_model.last_input_token_count
            self.total_output_token_count += self.tracked_model.last_output_token_count
            metrics_message += (
                f"| Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        metrics_message += "]"
        self.logger.log(Text(metrics_message, style="dim"), level=1)


class LogLevel(IntEnum):
    """
    Enumeration of logging verbosity levels.
    
    These levels control how much information is output during agent execution:
    - OFF: No output
    - ERROR: Only error messages
    - INFO: Standard information (default)
    - DEBUG: Detailed information for debugging
    """
    OFF = -1    # No output
    ERROR = 0   # Only errors 
    INFO = 1    # Normal output (default)
    DEBUG = 2   # Detailed output


# Color constant used for highlighting in log output
YELLOW_HEX = "#d4b702"


class AgentLogger:
    """
    Handles logging for agent activities with rich formatting.
    
    This class provides various methods for outputting different types of
    information (text, code, markdown) with consistent formatting and 
    respects the configured verbosity level.
    """
    
    def __init__(self, level: LogLevel = LogLevel.INFO):
        """
        Initialize a logger with the specified verbosity level.
        
        Args:
            level: The verbosity level (default: INFO)
        """
        self.level = level
        self.console = Console()

    def log(self, *args, level: str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """
        Log a message if the current verbosity level allows it.
        
        Args:
            *args: Content to log
            level: Minimum level required to display this message
            **kwargs: Additional arguments for rich.console.print
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def log_error(self, error_message: str) -> None:
        """
        Log an error message with appropriate styling.
        
        Args:
            error_message: The error message to display
        """
        self.log(escape_code_brackets(error_message), style="bold red", level=LogLevel.ERROR)

    def log_markdown(self, content: str, title: Optional[str] = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
        """
        Log content as markdown with optional title.
        
        Args:
            content: Markdown content to display
            title: Optional title to display above content
            level: Minimum level required to display this message
            style: Color style for the title
        """
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        if title:
            self.log(
                Group(
                    Rule(
                        "[bold italic]" + title,
                        align="left",
                        style=style,
                    ),
                    markdown_content,
                ),
                level=level,
            )
        else:
            self.log(markdown_content, level=level)

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        """
        Log content as syntax-highlighted code with a title.
        
        Args:
            title: Title to display above the code
            content: Code content to display with syntax highlighting
            level: Minimum level required to display this message
        """
        self.log(
            Panel(
                Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        """
        Log a horizontal rule with a title.
        
        Args:
            title: Title to display on the horizontal rule
            level: Minimum level required to display this message
        """
        self.log(
            Rule(
                "[bold]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=level,
        )

    def log_task(self, content: str, subtitle: str, title: Optional[str] = None, level: int = LogLevel.INFO) -> None:
        """
        Log task information in a panel with title and subtitle.
        
        Args:
            content: Task content to display
            subtitle: Subtitle for the panel
            title: Optional title for the panel (defaults to "Task")
            level: Minimum level required to display this message
        """
        title = title or "Task"
        task_panel = Panel(
            Text(content),
            title=f"[bold]{title}",
            subtitle=subtitle,
            subtitle_align="right",
            box=box.HORIZONTALS,
        )
        self.log(task_panel, level=level)

    def log_messages(self, messages: List, level: int = LogLevel.DEBUG) -> None:
        """
        Log a list of messages with role-based styling.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            level: Minimum level required to display these messages
        """
        for message in messages:
            if message["role"] == "system":
                self.log(message["content"], style="bold green", level=level)
            elif message["role"] == "user":
                self.log(message["content"], style="bold", level=level)
            else:
                self.log(message["content"], level=level)

    def visualize_agent_tree(self, agent):
        """
        Visualize the structure of an agent as a hierarchical tree.
        
        This displays the agent's tools and any managed sub-agents
        in a tree structure.
        
        Args:
            agent: The agent to visualize
        """
        def create_tools_section(tools_dict):
            """Create a tree section for the agent's tools."""
            tools_tree = Tree("Tools")
            for tool_name, tool in tools_dict.items():
                tool_node = tools_tree.add(f"[bold]{tool_name}[/bold]")
                if hasattr(tool, "description"):
                    tool_node.add(Text(tool.description, style="italic"))
            return tools_tree

        def get_agent_headline(agent, name: Optional[str] = None):
            """Generate the headline for an agent node."""
            return f"[bold]{name or agent.__class__.__name__}[/bold]"

        def build_agent_tree(parent_tree, agent_obj):
            """Recursively build the agent tree structure."""
            tools_tree = create_tools_section(agent_obj.tools_dict)
            parent_tree.add(tools_tree)
            
            if hasattr(agent_obj, "managed_agents") and agent_obj.managed_agents:
                managed_agents_tree = Tree("Managed Agents")
                for agent_name, managed_agent in agent_obj.managed_agents.items():
                    agent_node = managed_agents_tree.add(get_agent_headline(managed_agent, agent_name))
                    build_agent_tree(agent_node, managed_agent)
                parent_tree.add(managed_agents_tree)

        # Create the main agent tree and display it
        agent_tree = Tree(get_agent_headline(agent))
        build_agent_tree(agent_tree, agent)
        self.console.print(agent_tree)
