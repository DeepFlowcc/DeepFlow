"""
Gradio UI Module
==============

This module provides a user interface for the agent framework using Gradio.
It allows users to interact with agents through a web-based interface,
configure agent parameters, monitor agent execution, and visualize results.

The UI is designed to be intuitive and flexible, supporting different types of agents
and providing real-time feedback on agent actions and reasoning processes.

DeepFlow@2025
"""

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

# Standard library imports
import os           # For file system operations like path manipulation and directory creation
import re           # For regular expression operations used in text processing
import shutil       # For high-level file operations like copying
from typing import Optional  # For type hinting with optional parameters

# Import project-specific modules and types
from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types  # Agent data types
from smolagents.agents import ActionStep, MultiStepAgent  # Agent implementation components
from smolagents.memory import MemoryStep  # For tracking agent memory and steps
from smolagents.utils import _is_package_available  # Utility to check for dependencies


def pull_messages_from_step(
    step_log: MemoryStep,
):
    """
    Extract ChatMessage objects from agent steps with proper nesting.
    
    This function converts agent step logs into Gradio ChatMessage objects for display in the UI.
    It handles various types of content including model outputs, tool calls, errors, and execution logs.
    
    Args:
        step_log (MemoryStep): The memory step log containing agent actions and observations
        
    Yields:
        gradio.ChatMessage: Properly formatted chat messages for display in the Gradio UI
    """
    # Import gradio within the function to avoid requiring it unless this function is called
    import gradio as gr

    if isinstance(step_log, ActionStep):
        # Output the step number as a header message
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM if available
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output by removing whitespace
            model_output = step_log.model_output.strip()
            
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            # This ensures code blocks are properly formatted for display
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()
            
            # Create a chat message with the model's reasoning/thought process
            yield gr.ChatMessage(role="assistant", content=model_output)

        # Handle tool calls by creating nested messages
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            # Get the first tool call to determine how to format the display
            first_tool_call = step_log.tool_calls[0]
            # Check if the tool call is a Python code execution
            used_code = first_tool_call.name == "python_interpreter"
            # Create a unique ID for the parent message for nesting child messages
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Process the tool call arguments based on their type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                # If arguments are a dictionary, prefer the 'answer' field or stringify the dict
                content = str(args.get("answer", str(args)))
            else:
                # Otherwise, convert arguments to string
                content = str(args).strip()

            # Special formatting for Python code
            if used_code:
                # Clean up the content by removing any end code tags and format as Python code block
                content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                content = content.strip()
                # Ensure code is wrapped in Python code block markdown
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            # Create the parent message for the tool call
            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",  # Tool name as title
                    "id": parent_id,  # Unique ID for nesting
                    "status": "pending",  # Initial status is pending
                },
            )
            yield parent_message_tool

            # Add execution logs as a nested message under the tool call if available
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):  # Only yield execution logs if there's actual content
                log_content = step_log.observations.strip()
                if log_content:
                    # Remove prefix and format as a bash code block
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"```bash\n{log_content}\n",
                        metadata={"title": "📝 Execution Logs", "parent_id": parent_id, "status": "done"},
                    )

            # Add errors as a nested message under the tool call if available
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={"title": "💥 Error", "parent_id": parent_id, "status": "done"},
                )

            # Update parent message metadata to done status
            parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors (not from tool calls)
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        # Add a footnote with step number, token counts, and duration information
        step_footnote = f"{step_number}"
        # Add token information if available
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = (
                f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            )
            step_footnote += token_str
        # Add duration information if available
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        # Format the footnote as small, light gray text
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        # Add a separator line between steps
        yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """
    Runs an agent with the given task and streams the messages from the agent as Gradio ChatMessages.
    
    This function serves as a bridge between the agent execution and the Gradio UI,
    converting agent outputs into UI-friendly message formats in real-time.
    
    Args:
        agent: The agent to run
        task (str): The user's task or query
        reset_agent_memory (bool): Whether to reset the agent's memory before running
        additional_args (dict, optional): Additional arguments to pass to the agent
        
    Yields:
        gradio.ChatMessage: Chat messages for the Gradio UI representing agent actions and outputs
    """
    # Check if gradio is installed
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr

    # Initialize token counters
    total_input_tokens = 0
    total_output_tokens = 0

    # Run the agent with streaming enabled
    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track token usage if the model provides token counts
        if getattr(agent.model, "last_input_token_count", None) is not None:
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            # Store token counts in the step log for display
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        # Convert each step log to gradio chat messages and yield them
        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    # Process the final answer (the last step log)
    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)  # Ensure proper type wrapping

    # Format the final answer based on its type
    if isinstance(final_answer, AgentText):
        # Text answers are displayed as markdown
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        # Image answers are displayed as images with the proper MIME type
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        # Audio answers are displayed as audio players with the proper MIME type
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        # Other types are displayed as stringified values
        yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}")


class GradioUI:
    """
    A one-line interface to launch your agent in Gradio.
    
    This class provides an easy way to create a web-based user interface
    for interacting with agents, handling file uploads, and displaying results.
    """

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        """
        Initialize the Gradio UI with an agent and optional file upload support.
        
        Args:
            agent (MultiStepAgent): The agent to interact with through the UI
            file_upload_folder (str, optional): Path to store uploaded files
        """
        # Check if gradio is installed
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        
        # Store the agent
        self.agent = agent
        
        # Set up file upload folder if provided
        self.file_upload_folder = file_upload_folder
        
        # Get agent metadata for display
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        
        # Create the file upload folder if it doesn't exist
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages, session_state):
        """
        Handle user interaction with the agent.
        
        This function processes user input, runs the agent, and streams responses
        to the Gradio chat interface.
        
        Args:
            prompt (str): The user's input prompt
            messages (list): The current message history
            session_state (dict): State object to store session data
            
        Yields:
            list: Updated message history with agent responses
        """
        import gradio as gr

        # Initialize agent in session state if not already present
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            # Add user message to chat history
            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            # Stream agent responses and add to chat history
            for msg in stream_to_gradio(session_state["agent"], task=prompt, reset_agent_memory=False):
                messages.append(msg)
                yield messages

            yield messages
        except Exception as e:
            # Handle errors by displaying them in the chat
            print(f"Error in interaction: {str(e)}")
            messages.append(gr.ChatMessage(role="assistant", content=f"Error: {str(e)}"))
            yield messages

    def upload_file(self, file, file_uploads_log, allowed_file_types=None):
        """
        Handle file uploads and validate file types.
        
        Args:
            file: The uploaded file object
            file_uploads_log (list): Log of previously uploaded files
            allowed_file_types (list, optional): List of allowed file extensions
            
        Returns:
            tuple: Status textbox and updated file upload log
        """
        import gradio as gr

        # Handle case when no file is uploaded
        if file is None:
            return gr.Textbox(value="No file uploaded", visible=True), file_uploads_log

        # Set default allowed file types if none provided
        if allowed_file_types is None:
            allowed_file_types = [".pdf", ".docx", ".txt"]

        # Check if file extension is allowed
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name to ensure it's safe for the file system
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        # Return status and updated file log
        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        """
        Process user input and append uploaded file information if any.
        
        Args:
            text_input (str): The user's text input
            file_uploads_log (list): List of uploaded file paths
            
        Returns:
            tuple: Modified text input, empty string, and button with updated state
        """
        import gradio as gr

        # Append file information to the user's input if files have been uploaded
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",  # Clear the input box
            gr.Button(interactive=False),  # Disable the button during processing
        )

    def launch(self, share: bool = True, **kwargs):
        """
        Launch the Gradio UI for the agent.
        
        Args:
            share (bool): Whether to create a public link for the UI
            **kwargs: Additional arguments to pass to gradio.Blocks.launch()
        """
        import gradio as gr

        # Create the Gradio Blocks interface
        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Initialize session state and storage
            session_state = gr.State({})  # Stores session-specific data
            stored_messages = gr.State([])  # Stores message history
            file_uploads_log = gr.State([])  # Tracks uploaded files

            # Create the sidebar with agent information
            with gr.Sidebar():
                # Display agent name and description
                gr.Markdown(
                    f"# {self.name.replace('_', ' ').capitalize()}"
                    "\n> This web ui allows you to interact with a `smolagents` agent that can use tools and execute steps to complete tasks."
                    + (f"\n\n**Agent description:**\n{self.description}" if self.description else "")
                )

                # Create input components
                with gr.Group():
                    gr.Markdown("**Your request**", container=True)
                    # Text input for user prompts
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here and press Shift+Enter or press the button",
                    )
                    # Submit button
                    submit_btn = gr.Button("Submit", variant="primary")

                # Add file upload components if enabled
                if self.file_upload_folder is not None:
                    upload_file = gr.File(label="Upload a file")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                    # Set up file upload handling
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )

                # Add attribution footer
                gr.HTML("<br><br><h4><center>Powered by:</center></h4>")
                with gr.Row():
                    gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
            <a target="_blank" href="https://github.com/huggingface/smolagents"><b>huggingface/smolagents</b></a>
            </div>""")

            # Main chat interface
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,  # User avatar (none)
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",  # Agent avatar
                ),
                resizeable=True,
                scale=1,
            )

            # Set up event handlers for text input submission
            text_input.submit(
                self.log_user_message,  # First log the user message
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,  # Then interact with the agent
                [stored_messages, chatbot, session_state], 
                [chatbot]
            ).then(
                # Finally re-enable the input components
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            # Set up event handlers for button click (same flow as text submission)
            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent, 
                [stored_messages, chatbot, session_state], 
                [chatbot]
            ).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

        # Launch the Gradio interface
        demo.launch(debug=True, share=share, **kwargs)


# Export only the public API
__all__ = ["stream_to_gradio", "GradioUI"]
