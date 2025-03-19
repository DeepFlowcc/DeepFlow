"""
Agent Data Types Module
=====================

This module defines specialized data types that can be returned by agents.
These types provide consistent interfaces for handling different kinds of data
like text, images, and audio, ensuring proper serialization, deserialization,
and display capabilities.

DeepFlow@2025
"""

# Standard library imports
import logging            # For logging errors and information
import os                 # For filesystem operations
import pathlib            # For path manipulation with advanced features
import tempfile           # For creating temporary files and directories
import uuid               # For generating unique identifiers
from io import BytesIO    # For handling binary data in memory

# Third-party imports
import numpy as np                                # For numerical operations on arrays
import requests                                   # For HTTP requests
from huggingface_hub.utils import is_torch_available  # Check if PyTorch is installed
from PIL import Image                             # For image processing
from PIL.Image import Image as ImageType          # The PIL Image class type

# Local imports
from .utils import _is_package_available          # Utility to check if a package is available


# Set up module-level logger
logger = logging.getLogger(__name__)


class AgentType:
    """
    Abstract base class for agent-returned data types.
    
    This class defines the interface for all agent data types, enabling them to:
    1. Behave as their native type (string, image, etc.)
    2. Provide string representation via str(obj)
    3. Display correctly in interactive environments like Jupyter notebooks
    
    All specialized agent data types should inherit from this class.
    """

    def __init__(self, value):
        """
        Initialize with a native value.
        
        Args:
            value: The native value to wrap (string, image, etc.)
        """
        # Store the original value
        self._value = value

    def __str__(self):
        """
        Return string representation via the to_string() method.
        
        This allows using str(obj) on any AgentType object.
        """
        # Delegate to the to_string method which subclasses must implement
        return self.to_string()

    def to_raw(self):
        """
        Return the raw native value.
        
        This method should be overridden by subclasses to return their
        specific native value type.
        
        Returns:
            The native value in its original form.
        """
        # Log an error since this base method shouldn't be called directly
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        # Return the stored value anyway as fallback
        return self._value

    def to_string(self) -> str:
        """
        Convert the value to a string representation.
        
        This method should be overridden by subclasses to provide
        type-specific string conversion.
        
        Returns:
            str: String representation of the value.
        """
        # Log an error since this base method shouldn't be called directly
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        # Convert the value to string as fallback
        return str(self._value)


class AgentText(AgentType, str):
    """
    Text type returned by agents.
    
    Inherits from both AgentType and str to behave like a string
    while providing agent-specific capabilities.
    """

    def to_raw(self):
        """
        Return the raw string value.
        
        Returns:
            str: The original string value.
        """
        # Simply return the stored value
        return self._value

    def to_string(self):
        """
        Convert to string representation (already a string).
        
        Returns:
            str: The string representation.
        """
        # Convert the stored value to string using str()
        # This ensures we always return a string, even if _value isn't already a string
        return str(self._value)


class AgentImage(AgentType, ImageType):
    """
    Image type returned by agents.
    
    Inherits from both AgentType and PIL.Image to behave like a PIL image
    while providing agent-specific capabilities for serialization and display.
    """

    def __init__(self, value):
        """
        Initialize with an image value.
        
        Args:
            value: Can be a PIL.Image, AgentImage, file path, bytes, tensor, or array
        """
        # Initialize the parent AgentType class
        AgentType.__init__(self, value)
        # Initialize the PIL.Image class
        ImageType.__init__(self)

        # Initialize storage attributes to track different representations of the image
        self._image = None  # For PIL.Image representation
        self._path = None   # For file path representation
        self._tensor = None # For tensor representation

        # Handle different input types to populate the appropriate representation
        if isinstance(value, AgentImage):
            # Copy from another AgentImage instance
            self._image, self._path, self._tensor = value._image, value._path, value._tensor
        elif isinstance(value, ImageType):
            # Store PIL.Image directly
            self._image = value
        elif isinstance(value, bytes):
            # Convert bytes to PIL.Image using BytesIO
            self._image = Image.open(BytesIO(value))
        elif isinstance(value, (str, pathlib.Path)):
            # Store file path as string or Path
            self._path = value
        elif is_torch_available():
            # Only try to handle torch tensors if PyTorch is available
            import torch
            # Handle torch tensor
            if isinstance(value, torch.Tensor):
                self._tensor = value
            # Handle numpy array by converting to tensor
            if isinstance(value, np.ndarray):
                self._tensor = torch.from_numpy(value)

        # Ensure at least one representation is available
        if self._path is None and self._image is None and self._tensor is None:
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        Display the image in Jupyter notebooks.
        
        This method is called automatically by IPython display mechanics
        when the object is the result of a cell execution.
        
        Args:
            include: Parameters to include (unused, for compatibility)
            exclude: Parameters to exclude (unused, for compatibility)
        """
        # Import IPython display utilities here to avoid global dependency
        from IPython.display import Image, display

        # Display the image using IPython's display function
        # Convert to string first, which gives the file path
        display(Image(self.to_string()))

    def to_raw(self):
        """
        Return the raw PIL.Image representation.
        
        Converts from any available representation (file path, tensor)
        to a PIL.Image if needed.
        
        Returns:
            PIL.Image: The image as a PIL Image object
        """
        # If we already have the image in memory, return it
        if self._image is not None:
            return self._image

        # If we have a file path, load the image from disk
        if self._path is not None:
            # Load from file path and cache the result
            self._image = Image.open(self._path)
            return self._image

        # If we have a tensor, convert it to a PIL.Image
        if self._tensor is not None:
            # Convert tensor to numpy array on CPU
            array = self._tensor.cpu().detach().numpy()
            # Convert array to PIL.Image with pixel value scaling and type conversion
            return Image.fromarray((255 - array * 255).astype(np.uint8))
            
        # If no representation is available, this will not be reached due to initialization check

    def to_string(self):
        """
        Return a file path to the image.
        
        If the image is not already saved to a file, saves it to a temporary
        location and returns that path.
        
        Returns:
            str: A file path to the image
        """
        # If we already have a file path, return it
        if self._path is not None:
            return self._path

        # If we have a PIL.Image, save it to a temporary file
        if self._image is not None:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            # Generate a unique filename
            self._path = os.path.join(temp_dir, str(uuid.uuid4()) + ".png")
            # Save the image to the temporary path
            self._image.save(self._path, format="png")
            return self._path

        # If we have a tensor, convert it to an image and save
        if self._tensor is not None:
            # Convert tensor to numpy array on CPU
            array = self._tensor.cpu().detach().numpy()
            # Convert array to PIL.Image
            img = Image.fromarray((255 - array * 255).astype(np.uint8))
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            # Generate a unique filename
            self._path = os.path.join(temp_dir, str(uuid.uuid4()) + ".png")
            # Save the image to the temporary path
            img.save(self._path, format="png")
            return self._path
            
        # If no representation is available, this will not be reached due to initialization check

    def save(self, output_path, format: str = None, **params):
        """
        Save the image to a file.
        
        Args:
            output_path: Path where the image should be saved
            format: Image format (e.g. 'png', 'jpg')
            **params: Additional parameters for PIL.Image.save
        """
        # Get the raw PIL.Image representation
        img = self.to_raw()
        # Save the image to the specified path with given format and parameters
        img.save(output_path, format=format, **params)


class AgentAudio(AgentType, str):
    """
    Audio type returned by agents.
    
    Provides functionality for handling audio data with consistent
    interfaces for serialization and playback.
    """

    def __init__(self, value, sample_rate=16_000):
        """
        Initialize with audio value.
        
        Args:
            value: Can be an AgentAudio, file path, bytes, or numpy array
            sample_rate: Sample rate of the audio in Hz (default: 16000)
        """
        # Initialize the parent AgentType class
        AgentType.__init__(self, value)
        
        # Initialize storage attributes for different representations
        self._audio = None  # For numpy array data
        self._path = None   # For file path
        self._sample_rate = sample_rate  # Store the sample rate
        
        # Handle different input types
        if isinstance(value, AgentAudio):
            # Copy from another AgentAudio instance
            self._audio, self._path, self._sample_rate = value._audio, value._path, value._sample_rate
        elif isinstance(value, (str, pathlib.Path)):
            # Store file path as string
            self._path = str(value)
        elif isinstance(value, bytes):
            # Save bytes to a temporary file
            temp_dir = tempfile.mkdtemp()
            self._path = os.path.join(temp_dir, str(uuid.uuid4()) + ".wav")
            # Write the bytes to the file
            with open(self._path, "wb") as f:
                f.write(value)
        elif _is_package_available("numpy") and isinstance(value, np.ndarray):
            # Store numpy array directly
            self._audio = value
        else:
            # Raise an error for unsupported types
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        Display the audio in Jupyter notebooks with playback controls.
        
        This method is called automatically by IPython display mechanics
        when the object is the result of a cell execution.
        
        Args:
            include: Parameters to include (unused, for compatibility)
            exclude: Parameters to exclude (unused, for compatibility)
        """
        # Import IPython display utilities here to avoid global dependency
        from IPython.display import Audio, display

        # Display the audio with the appropriate sample rate
        display(Audio(self.to_string(), rate=self._sample_rate))

    def to_raw(self):
        """
        Return the raw numpy array representation of the audio.
        
        Converts from any available representation (file path) to
        a numpy array if needed.
        
        Returns:
            numpy.ndarray: The audio data as a numpy array
            
        Raises:
            ValueError: If audio cannot be converted to raw format
        """
        # If we already have the audio array in memory, return it
        if self._audio is not None:
            return self._audio

        # If we have a file path, load the audio from disk
        if self._path is not None:
            # Load the audio file using soundfile
            import soundfile as sf

            # Read the audio file and get both data and sample rate
            self._audio, self._sample_rate = sf.read(self._path)
            return self._audio

        # If no valid representation is available
        raise ValueError("Could not convert audio to raw format")

    def to_string(self):
        """
        Return a file path to the audio.
        
        If the audio is not already saved to a file, saves it to a temporary
        location and returns that path.
        
        Returns:
            str: A file path to the audio
            
        Raises:
            ValueError: If audio cannot be converted to string format
        """
        # If we already have a file path, return it
        if self._path is not None:
            return self._path

        # If we have a numpy array, save it to a temporary file
        if self._audio is not None:
            # Check if soundfile is available
            if not _is_package_available("soundfile"):
                raise ImportError(
                    "The soundfile package is required to save audio files. Please install it with `pip install soundfile`."
                )
            
            # Import soundfile for audio file handling
            import soundfile as sf

            # Create a temporary directory and file
            temp_dir = tempfile.mkdtemp()
            self._path = os.path.join(temp_dir, str(uuid.uuid4()) + ".wav")
            
            # Write the audio data to the file
            sf.write(self._path, self._audio, self._sample_rate)
            return self._path

        # If no valid representation is available
        raise ValueError("Could not convert audio to string format")


# Mapping of type names to agent type classes for type conversion
_AGENT_TYPE_MAPPING = {"string": AgentText, "image": AgentImage, "audio": AgentAudio}


def handle_agent_input_types(*args, **kwargs):
    """
    Normalize agent inputs to their raw formats.
    
    This function prepares input arguments for processing by tools or models
    by converting agent types to their raw native values.
    
    Args:
        *args: Positional arguments to normalize
        **kwargs: Keyword arguments to normalize
    
    Returns:
        tuple: A tuple containing (args, kwargs) with normalized values
    """
    # Currently a pass-through function - the raw types are used directly
    # This is a placeholder for potential future pre-processing
    return args, kwargs


def handle_agent_output_types(output, output_type=None):
    """
    Normalize outputs to expected agent types.
    
    This function ensures that outputs from tools or models are properly
    wrapped in appropriate agent types based on the expected output_type.
    
    Args:
        output: The raw output value to process
        output_type: Expected output type string ('image', 'audio', 'string')
    
    Returns:
        An agent type wrapped value or the original output if no wrapping is needed
    """
    # If output_type is specified, wrap the output in the appropriate agent type
    if output_type is not None and output_type == "image" and not isinstance(output, AgentImage):
        # Wrap output in AgentImage if expected type is 'image'
        return AgentImage(output)
    elif output_type is not None and output_type == "audio" and not isinstance(output, AgentAudio):
        # Wrap output in AgentAudio if expected type is 'audio'
        return AgentAudio(output)
    elif output_type is not None and output_type == "string" and not isinstance(output, AgentText):
        # Wrap output in AgentText if expected type is 'string'
        return AgentText(output)
    
    # Return the original output if no wrapping is needed or type is unknown
    return output


# List of classes to expose in the public API
__all__ = ["AgentType", "AgentImage", "AgentText", "AgentAudio"]
