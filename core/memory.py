"""
Agent Memory Module
=================

This module implements the memory system for agents, providing structured storage for
different types of execution steps, actions, plans, and observations. The memory system
enables agents to track their reasoning process, store intermediate results, and 
maintain context across multi-step executions.

The module defines various types of memory steps (TaskStep, ActionStep, PlanningStep)
and a centralized AgentMemory class that manages these steps and provides methods
for querying and manipulating agent memory.

DeepFlow@2025
"""

from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict, Union

from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


logger = getLogger(__name__)


class Message(TypedDict):
    """
    Typed dictionary for representing message data.
    
    This structure is used for standardized communication between
    the agent and its memory system.
    
    Attributes:
        role: The role of the message sender (user, assistant, system, etc.)
        content: The content of the message (text or structured content)
    """
    role: MessageRole
    content: str | list[dict]


@dataclass
class ToolCall:
    """
    Represents a single call to a tool by the agent.
    
    This dataclass stores information about a specific tool invocation,
    including the tool name, arguments passed, and a unique ID for tracking.
    
    Attributes:
        name: The name of the tool being called
        arguments: The arguments passed to the tool
        id: A unique identifier for this specific tool call
    """
    name: str
    arguments: Any
    id: str

    def to_dict(self):
        """
        Convert the tool call to a dictionary representation.
        
        Returns:
            dict: A serialized representation of the tool call
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    """
    Base class for all agent memory steps.
    
    This abstract class defines the interface that all memory step types
    must implement, providing serialization and message conversion capabilities.
    """
    
    def to_dict(self):
        """
        Convert the memory step to a dictionary representation.
        
        Returns:
            dict: A serialized representation of the memory step
        """
        return asdict(self)

    def to_messages(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Convert the memory step to a list of messages.
        
        This abstract method should be implemented by subclasses to convert
        their specific data into standardized messages.
        
        Args:
            **kwargs: Additional arguments for conversion process
            
        Returns:
            List[Message]: A list of messages representing this memory step
        """
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    """
    Memory step representing an agent action.
    
    This step records the execution of a single action by the agent,
    including inputs, outputs, observations, and any errors encountered.
    
    Attributes:
        model_input_messages: Messages provided to the model
        tool_calls: Tools called during this step
        start_time: When the step began execution
        end_time: When the step completed execution
        step_number: Sequential number of this step 
        error: Any error encountered during execution
        duration: Time taken to execute the step
        model_output_message: The full message output by the model
        model_output: The text output by the model
        observations: Text observations from tool execution
        observations_images: Image observations from tool execution
        action_output: Final output from the action
    """
    model_input_messages: List[Message] | None = None
    tool_calls: List[ToolCall] | None = None
    start_time: float | None = None
    end_time: float | None = None
    step_number: int | None = None
    error: AgentError | None = None
    duration: float | None = None
    model_output_message: ChatMessage = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: List["PIL.Image.Image"] | None = None
    action_output: Any = None

    def to_dict(self):
        """
        Convert the action step to a dictionary representation.
        
        Returns:
            dict: A serialized representation of the action step
        """
        return {
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls] if self.tool_calls else [],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "step": self.step_number,
            "error": self.error.to_dict() if self.error else None,
            "duration": self.duration,
            "model_output_message": self.model_output_message,
            "model_output": self.model_output,
            "observations": self.observations,
            "action_output": make_json_serializable(self.action_output),
        }

    def to_messages(self, summary_mode: bool = False, show_model_input_messages: bool = False) -> List[Message]:
        """
        Convert the action step to a list of messages.
        
        Args:
            summary_mode: Whether to generate a summarized version
            show_model_input_messages: Whether to include input messages
            
        Returns:
            List[Message]: Messages representing this action step
        """
        messages = []
        # Include input messages if requested
        if self.model_input_messages is not None and show_model_input_messages:
            messages.append(Message(role=MessageRole.SYSTEM, content=self.model_input_messages))
        
        # Include model output unless in summary mode
        if self.model_output is not None and not summary_mode:
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        # Include tool calls if present
        if self.tool_calls is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.to_dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        # Include observations if present
        if self.observations is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": (f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else "")
                            + f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
            
        # Include error information if present
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                Message(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        # Include image observations if present
        if self.observations_images:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": "Here are the observed images:"}]
                    + [
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )
        return messages


@dataclass
class PlanningStep(MemoryStep):
    """
    Memory step representing a planning operation.
    
    This step records the input, output, and results of a planning operation,
    where the agent analyzes the current state and formulates a plan.
    
    Attributes:
        model_input_messages: Messages provided to the model
        model_output_message_facts: The model's output message containing facts
        facts: The gathered facts as text
        model_output_message_plan: The model's output message containing the plan
        plan: The formulated plan as text
    """
    model_input_messages: List[Message]
    model_output_message_facts: ChatMessage
    facts: str
    model_output_message_plan: ChatMessage
    plan: str

    def to_messages(self, summary_mode: bool, **kwargs) -> List[Message]:
        """
        Convert the planning step to a list of messages.
        
        Args:
            summary_mode: Whether to generate a summarized version
            **kwargs: Additional conversion arguments
            
        Returns:
            List[Message]: Messages representing this planning step
        """
        messages = []
        # Always include facts
        messages.append(
            Message(
                role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[FACTS LIST]:\n{self.facts.strip()}"}]
            )
        )

        # Include plan unless in summary mode
        if not summary_mode:
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[PLAN]:\n{self.plan.strip()}"}]
                )
            )
        return messages


@dataclass
class TaskStep(MemoryStep):
    """
    Memory step representing a task assignment.
    
    This step records a task that has been assigned to the agent,
    including any associated images.
    
    Attributes:
        task: The task description as text
        task_images: Optional list of images associated with the task
    """
    task: str
    task_images: List["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        """
        Convert the task step to a list of messages.
        
        Args:
            summary_mode: Whether to generate a summarized version
            **kwargs: Additional conversion arguments
            
        Returns:
            List[Message]: Messages representing this task step
        """
        # Create content with task text
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        
        # Add images if present
        if self.task_images:
            for image in self.task_images:
                content.append({"type": "image", "image": image})

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    """
    Memory step representing a system prompt.
    
    This step records a system prompt that provides context and
    instructions to the agent.
    
    Attributes:
        system_prompt: The system prompt text
    """
    system_prompt: str

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        """
        Convert the system prompt step to a list of messages.
        
        Args:
            summary_mode: Whether to generate a summarized version
            **kwargs: Additional conversion arguments
            
        Returns:
            List[Message]: Messages representing this system prompt step
        """
        # Skip in summary mode
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


class AgentMemory:
    """
    Central management system for agent memory.
    
    This class stores, organizes, and provides access to all memory steps
    that an agent accumulates during execution, enabling context retention
    and analysis of the agent's execution history.
    """
    
    def __init__(self, system_prompt: str = ""):
        """
        Initialize an agent memory instance.
        
        Args:
            system_prompt: Optional system prompt to initialize with
        """
        self.steps = []
        if system_prompt:
            self.add_system_prompt(system_prompt)

    def reset(self):
        """
        Clear all stored memory steps.
        """
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """
        Get a concise representation of all memory steps.
        
        Returns a list of dictionaries without the model input messages,
        which can be verbose and less relevant for succinct histories.
        
        Returns:
            list[dict]: Concise representation of all memory steps
        """
        return [
            {key: value for key, value in step.to_dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """
        Get a complete representation of all memory steps.
        
        Returns:
            list[dict]: Complete representation of all memory steps
        """
        return [step.to_dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """
        Replay the agent's execution history through the logger.
        
        This method outputs all steps in the agent's memory to the provided
        logger for analysis or debugging.
        
        Args:
            logger: The logger to output the replay
            detailed: Whether to include detailed information
        """
        for i, step in enumerate(self.steps):
            if isinstance(step, ActionStep):
                # Log action step components
                if step.model_output is not None:
                    logger.log(f"Step {i}: Model output:\n{step.model_output}", LogLevel.DEBUG if detailed else LogLevel.INFO)
                if step.tool_calls is not None:
                    tool_calls_str = ", ".join([f"{tc.name}({tc.arguments})" for tc in step.tool_calls])
                    logger.log(f"Step {i}: Tool calls: {tool_calls_str}", LogLevel.DEBUG if detailed else LogLevel.INFO)
                if step.observations is not None:
                    logger.log(f"Step {i}: Observations: {step.observations}", LogLevel.DEBUG if detailed else LogLevel.INFO)
                if step.error is not None:
                    logger.log(f"Step {i}: Error: {step.error}", LogLevel.DEBUG if detailed else LogLevel.INFO)
                if step.action_output is not None:
                    logger.log(f"Step {i}: Action output: {step.action_output}", LogLevel.DEBUG if detailed else LogLevel.INFO)
            elif isinstance(step, PlanningStep):
                # Log planning step components
                logger.log(f"Step {i}: Planning step", LogLevel.DEBUG if detailed else LogLevel.INFO)
                logger.log(f"Facts: {step.facts}", LogLevel.DEBUG)
                logger.log(f"Plan: {step.plan}", LogLevel.DEBUG)
            elif isinstance(step, TaskStep):
                # Log task step
                logger.log(f"Step {i}: Task: {step.task}", LogLevel.DEBUG if detailed else LogLevel.INFO)

    def add_action_step(self, action_step: ActionStep):
        """
        Add an action step to memory.
        
        Args:
            action_step: The action step to add
        """
        self.steps.append(action_step)

    def add_planning_step(self, planning_step: PlanningStep):
        """
        Add a planning step to memory.
        
        Args:
            planning_step: The planning step to add
        """
        self.steps.append(planning_step)

    def add_task(self, task: str, task_images: List["PIL.Image.Image"] | None = None):
        """
        Add a task step to memory.
        
        Args:
            task: The task description
            task_images: Optional images associated with the task
        """
        self.steps.append(TaskStep(task=task, task_images=task_images))

    def add_system_prompt(self, system_prompt: str):
        """
        Add a system prompt step to memory.
        
        Args:
            system_prompt: The system prompt text
        """
        self.steps.append(SystemPromptStep(system_prompt=system_prompt))

    def get_messages(self, summary_mode: bool = False, show_model_input_messages: bool = False) -> List[Message]:
        """
        Convert all memory steps to a consolidated list of messages.
        
        This method aggregates messages from all memory steps to provide
        a complete conversation history.
        
        Args:
            summary_mode: Whether to generate summarized messages
            show_model_input_messages: Whether to include input messages
            
        Returns:
            List[Message]: Consolidated list of messages from all steps
        """
        messages = []
        for step in self.steps:
            messages.extend(step.to_messages(summary_mode=summary_mode, show_model_input_messages=show_model_input_messages))
        return messages


__all__ = ["AgentMemory"]
