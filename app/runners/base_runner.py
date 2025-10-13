"""
Base abstract class for model runners.
Defines the common interface that all framework-specific runners must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """
    Abstract base class for model runners.
    All framework-specific runners must inherit from this class and implement its methods.
    """
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.framework_name = ""
    
    @abstractmethod
    def can_run_model(self, model_path: str) -> bool:
        """
        Check if this runner can handle the given model file.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if this runner can handle the model, False otherwise
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """
        Load the model from the given path.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model (input/output shapes, types, etc.).
        
        Returns:
            Dict[str, Any]: Model information including input/output specifications
        """
        pass
    
    @abstractmethod
    def generate_sample_input(self) -> Any:
        """
        Generate a sample input for the model based on its input specifications.
        
        Returns:
            Any: Sample input data that matches the model's expected input format
        """
        pass
    
    @abstractmethod
    def run_inference(self, sample_input: Any) -> Any:
        """
        Run inference on the model with the given input.
        
        Args:
            sample_input (Any): Input data for inference
            
        Returns:
            Any: Model output
        """
        pass
    
    @abstractmethod
    def validate_output(self, output: Any, expected_shapes: Optional[List] = None) -> Dict[str, Any]:
        """
        Validate the model output to ensure it's in the expected format.
        
        Args:
            output (Any): Model output to validate
            expected_shapes (Optional[List]): Expected output shapes (if known)
            
        Returns:
            Dict[str, Any]: Validation results including success status and details
        """
        pass
    
    def run_full_test(self, model_path: str) -> Dict[str, Any]:
        """
        Run a complete test of the model: load, get info, generate input, run inference, validate output.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            Dict[str, Any]: Complete test results
        """
        results = {
            "framework": self.framework_name,
            "success": False,
            "steps": {},
            "error": None,
            "error_category": None,
            "user_friendly_message": None,
            "suggestions": []
        }
        
        try:
            # Step 1: Check if we can run this model
            if not self.can_run_model(model_path):
                results["error"] = f"{self.framework_name} runner cannot handle this model type"
                return results
            
            # Step 2: Load the model
            logger.info(f"Loading {self.framework_name} model from {model_path}")
            if not self.load_model(model_path):
                results["error"] = f"Failed to load {self.framework_name} model"
                return results
            results["steps"]["load_model"] = True
            
            # Step 3: Get model information
            model_info = self.get_model_info()
            results["steps"]["get_model_info"] = model_info
            
            # Step 4: Generate sample input
            sample_input = self.generate_sample_input()
            results["steps"]["sample_input"] = str(type(sample_input))
            
            # Step 5: Run inference
            output = self.run_inference(sample_input)
            results["steps"]["inference_output"] = str(type(output))
            
            # Step 6: Validate output
            validation = self.validate_output(output)
            results["steps"]["validation"] = validation
            
            # Overall success if all steps completed and validation passed
            results["success"] = validation.get("valid", False)
            
        except Exception as e:
            logger.error(f"Error during {self.framework_name} model testing: {str(e)}")
            error_info = self._categorize_error(str(e))
            results["error"] = str(e)
            results["error_category"] = error_info["category"]
            results["user_friendly_message"] = error_info["message"]
            results["suggestions"] = error_info["suggestions"]
        
        return results
    
    def _categorize_error(self, error_message: str) -> Dict[str, Any]:
        """
        Categorize errors and provide user-friendly messages and suggestions.
        
        Args:
            error_message (str): The error message
            
        Returns:
            Dict[str, Any]: Error categorization with user-friendly message and suggestions
        """
        error_lower = error_message.lower()
        
        # Version compatibility errors
        if any(keyword in error_lower for keyword in ["ir version", "version", "unsupported", "max supported"]):
            return {
                "category": "version_compatibility",
                "message": "Model version is not compatible with the current runtime",
                "suggestions": [
                    "Try converting your model to a compatible version",
                    "Check if you're using the latest model format",
                    "Consider using a different model export tool",
                    "Contact support if the issue persists"
                ]
            }
        
        # Dependency/installation errors
        if any(keyword in error_lower for keyword in ["import", "module", "not found", "install", "dependency"]):
            return {
                "category": "dependency_missing",
                "message": "Required dependencies are missing or not properly installed",
                "suggestions": [
                    "Install the required framework dependencies",
                    "Check your Python environment setup",
                    "Verify all packages are correctly installed",
                    "Try recreating your virtual environment"
                ]
            }
        
        # Model format/corruption errors
        if any(keyword in error_lower for keyword in ["corrupt", "invalid", "format", "parse", "decode"]):
            return {
                "category": "model_format",
                "message": "Model file appears to be corrupted or in an invalid format",
                "suggestions": [
                    "Verify the model file is not corrupted",
                    "Try re-exporting your model",
                    "Check if the file was properly uploaded",
                    "Ensure the model format is correct"
                ]
            }
        
        # Runtime/memory errors
        if any(keyword in error_lower for keyword in ["memory", "out of memory", "resource", "timeout"]):
            return {
                "category": "runtime_resource",
                "message": "Model execution failed due to resource constraints",
                "suggestions": [
                    "Try with a smaller model or reduced input size",
                    "Check available system memory",
                    "Consider model optimization or quantization",
                    "Contact support for resource limits"
                ]
            }
        
        # Input/output shape errors
        if any(keyword in error_lower for keyword in ["shape", "dimension", "size mismatch", "input", "output"]):
            return {
                "category": "input_output_mismatch",
                "message": "Model input/output shapes don't match expectations",
                "suggestions": [
                    "Verify the model's expected input/output format",
                    "Check your input data dimensions",
                    "Ensure the model was trained for your use case",
                    "Review the model documentation"
                ]
            }
        
        # Generic error fallback
        return {
            "category": "unknown",
            "message": "An unexpected error occurred during model execution",
            "suggestions": [
                "Check the model file and try again",
                "Verify all requirements are met",
                "Contact support with error details",
                "Try with a different model"
            ]
        }
    
    def cleanup(self):
        """Clean up resources used by the runner."""
        self.model = None
        self.model_loaded = False
