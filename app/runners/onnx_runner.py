"""
ONNX model runner implementation.
Handles loading, execution, and validation of ONNX models.
"""

import os
import tempfile
import zipfile
import io
from typing import Any, Dict, List, Optional
import numpy as np
import logging

try:
    import onnxruntime as ort
    import onnx
except ImportError:
    ort = None
    onnx = None

from .base_runner import BaseRunner

logger = logging.getLogger(__name__)


class ONNXRunner(BaseRunner):
    """
    ONNX model runner that can load, execute, and validate ONNX models.
    """
    
    def __init__(self):
        super().__init__()
        self.framework_name = "ONNX"
        
        if ort is None or onnx is None:
            logger.warning("ONNX Runtime or ONNX not installed. ONNX runner will not be available.")
    
    def can_run_model(self, model_path: str) -> bool:
        """
        Check if this runner can handle the given model file.
        """
        if ort is None or onnx is None:
            return False
            
        # Check if it's a .onnx file
        if model_path.lower().endswith('.onnx'):
            return True
            
        # Check if it's a zip file containing .onnx files
        if model_path.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    return any(f.lower().endswith('.onnx') for f in file_list)
            except:
                return False
        
        return False
    
    def _extract_onnx_from_zip(self, zip_path: str) -> str:
        """
        Extract ONNX model from zip file to a temporary file.
        
        Args:
            zip_path (str): Path to the zip file
            
        Returns:
            str: Path to the extracted ONNX file
        """
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Find the first .onnx file
            onnx_file = None
            for file in file_list:
                if file.lower().endswith('.onnx'):
                    onnx_file = file
                    break
            
            if not onnx_file:
                raise ValueError("No ONNX file found in zip")
            
            # Extract to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx') as temp_file:
                temp_file.write(zip_ref.read(onnx_file))
                return temp_file.name
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the ONNX model from the given path.
        """
        try:
            if ort is None or onnx is None:
                logger.error("ONNX Runtime not available")
                return False
            
            # Handle zip files
            if model_path.lower().endswith('.zip'):
                model_path = self._extract_onnx_from_zip(model_path)
            
            # Load ONNX model for inspection
            self.onnx_model = onnx.load(model_path)
            
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(model_path)
            
            self.model_loaded = True
            self.model_path = model_path
            logger.info(f"Successfully loaded ONNX model: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            # Re-raise with more context for better error categorization
            if "IR version" in str(e) and "max supported" in str(e):
                raise Exception(f"ONNX IR version compatibility error: {str(e)}")
            elif "onnx" in str(e).lower() and "not found" in str(e).lower():
                raise Exception(f"ONNX dependency missing: {str(e)}")
            else:
                raise Exception(f"ONNX model loading failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded ONNX model.
        """
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Get input information
            inputs = []
            for input_meta in self.session.get_inputs():
                input_info = {
                    "name": input_meta.name,
                    "shape": input_meta.shape,
                    "type": input_meta.type
                }
                inputs.append(input_info)
            
            # Get output information
            outputs = []
            for output_meta in self.session.get_outputs():
                output_info = {
                    "name": output_meta.name,
                    "shape": output_meta.shape,
                    "type": output_meta.type
                }
                outputs.append(output_info)
            
            return {
                "inputs": inputs,
                "outputs": outputs,
                "model_size_mb": os.path.getsize(self.model_path) / (1024 * 1024) if hasattr(self, 'model_path') else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
    
    def generate_sample_input(self) -> Dict[str, np.ndarray]:
        """
        Generate sample input for the ONNX model.
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded")
        
        sample_inputs = {}
        
        for input_meta in self.session.get_inputs():
            input_name = input_meta.name
            input_shape = input_meta.shape
            
            # Handle dynamic shapes by replacing with reasonable defaults
            actual_shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim == -1:
                    # Use common defaults for dynamic dimensions
                    if dim == 'batch_size' or dim == -1:
                        actual_shape.append(1)  # Batch size of 1
                    else:
                        actual_shape.append(32)  # Default size
                else:
                    actual_shape.append(dim)
            
            # Generate random input based on type
            if 'float' in input_meta.type:
                sample_inputs[input_name] = np.random.randn(*actual_shape).astype(np.float32)
            elif 'int' in input_meta.type:
                sample_inputs[input_name] = np.random.randint(0, 10, actual_shape).astype(np.int32)
            else:
                # Default to float32
                sample_inputs[input_name] = np.random.randn(*actual_shape).astype(np.float32)
        
        return sample_inputs
    
    def run_inference(self, sample_input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on the ONNX model.
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded")
        
        try:
            # Run inference
            outputs = self.session.run(None, sample_input)
            
            # Convert to dictionary format
            output_names = [output.name for output in self.session.get_outputs()]
            output_dict = {}
            
            for i, output_name in enumerate(output_names):
                output_dict[output_name] = outputs[i]
            
            return output_dict
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise
    
    def validate_output(self, output: Dict[str, np.ndarray], expected_shapes: Optional[List] = None) -> Dict[str, Any]:
        """
        Validate the model output.
        """
        validation_result = {
            "valid": True,
            "details": {},
            "warnings": []
        }
        
        try:
            # Check if output is a dictionary
            if not isinstance(output, dict):
                validation_result["valid"] = False
                validation_result["details"]["error"] = "Output is not a dictionary"
                return validation_result
            
            # Validate each output
            for output_name, output_data in output.items():
                if not isinstance(output_data, np.ndarray):
                    validation_result["valid"] = False
                    validation_result["details"][output_name] = "Output is not a numpy array"
                    continue
                
                # Check for NaN or Inf values
                if np.any(np.isnan(output_data)):
                    validation_result["warnings"].append(f"{output_name} contains NaN values")
                
                if np.any(np.isinf(output_data)):
                    validation_result["warnings"].append(f"{output_name} contains infinite values")
                
                # Store output info
                validation_result["details"][output_name] = {
                    "shape": output_data.shape,
                    "dtype": str(output_data.dtype),
                    "min": float(np.min(output_data)),
                    "max": float(np.max(output_data)),
                    "mean": float(np.mean(output_data))
                }
            
            # Check if we have outputs
            if not output:
                validation_result["valid"] = False
                validation_result["details"]["error"] = "No outputs produced"
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["details"]["error"] = str(e)
        
        return validation_result
    
    def cleanup(self):
        """Clean up resources."""
        super().cleanup()
        if hasattr(self, 'session'):
            del self.session
        if hasattr(self, 'onnx_model'):
            del self.onnx_model
