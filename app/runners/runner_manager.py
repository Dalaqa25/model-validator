"""
Runner manager that handles model execution testing for different frameworks.
"""

import tempfile
import os
import logging
from typing import Dict, Any, List, Optional
from .base_runner import BaseRunner
from .onnx_runner import ONNXRunner

logger = logging.getLogger(__name__)


class RunnerManager:
    """
    Manages different model runners and executes model testing.
    """
    
    def __init__(self):
        self.runners: List[BaseRunner] = [
            ONNXRunner(),
            # Add more runners here as they're implemented
        ]
        self.active_runners = [runner for runner in self.runners if runner.framework_name != ""]
    
    def find_compatible_runner(self, model_path: str) -> Optional[BaseRunner]:
        """
        Find a runner that can handle the given model.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            Optional[BaseRunner]: Compatible runner or None if none found
        """
        for runner in self.active_runners:
            if runner.can_run_model(model_path):
                return runner
        return None
    
    def test_model_execution(self, zip_contents: bytes, framework_used: str) -> Dict[str, Any]:
        """
        Test if a model can actually run by executing it with sample inputs.
        
        Args:
            zip_contents (bytes): ZIP file contents containing the model
            framework_used (str): Detected framework from validation
            
        Returns:
            Dict[str, Any]: Execution test results
        """
        results = {
            "execution_test": {
                "success": False,
                "runner_used": None,
                "framework_matched": False,
                "test_results": None,
                "error": None,
                "error_category": None,
                "user_friendly_message": None,
                "suggestions": []
            }
        }
        
        try:
            # Create temporary file for the zip
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                temp_file.write(zip_contents)
                temp_zip_path = temp_file.name
            
            try:
                # Find compatible runner
                compatible_runner = self.find_compatible_runner(temp_zip_path)
                
                if not compatible_runner:
                    results["execution_test"]["error"] = f"No compatible runner found for framework: {framework_used}"
                    return results
                
                # Check if framework matches
                framework_matched = (
                    compatible_runner.framework_name.lower() in framework_used.lower() or
                    framework_used.lower() in compatible_runner.framework_name.lower()
                )
                results["execution_test"]["framework_matched"] = framework_matched
                
                # Run the full test
                logger.info(f"Testing model execution with {compatible_runner.framework_name} runner")
                test_results = compatible_runner.run_full_test(temp_zip_path)
                
                results["execution_test"]["runner_used"] = compatible_runner.framework_name
                results["execution_test"]["test_results"] = test_results
                results["execution_test"]["success"] = test_results.get("success", False)
                
                # Pass through error information if test failed
                if not test_results.get("success", False):
                    results["execution_test"]["error"] = test_results.get("error")
                    results["execution_test"]["error_category"] = test_results.get("error_category")
                    results["execution_test"]["user_friendly_message"] = test_results.get("user_friendly_message")
                    results["execution_test"]["suggestions"] = test_results.get("suggestions", [])
                
                # Cleanup
                compatible_runner.cleanup()
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_zip_path):
                    os.unlink(temp_zip_path)
                
        except Exception as e:
            logger.error(f"Error during model execution testing: {str(e)}")
            results["execution_test"]["error"] = str(e)
        
        return results
    
    def get_available_runners(self) -> List[str]:
        """
        Get list of available runner framework names.
        
        Returns:
            List[str]: List of available framework names
        """
        return [runner.framework_name for runner in self.active_runners]
    
    def get_execution_summary(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a clean summary of execution results for frontend consumption.
        
        Args:
            execution_results (Dict[str, Any]): Results from test_model_execution
            
        Returns:
            Dict[str, Any]: Clean summary for frontend
        """
        execution_test = execution_results.get("execution_test", {})
        
        summary = {
            "execution_successful": execution_test.get("success", False),
            "runner_used": execution_test.get("runner_used"),
            "framework_matched": execution_test.get("framework_matched", False),
            "error_info": None
        }
        
        # Add error information if execution failed
        if not execution_test.get("success", False):
            summary["error_info"] = {
                "category": execution_test.get("error_category"),
                "message": execution_test.get("user_friendly_message"),
                "suggestions": execution_test.get("suggestions", []),
                "technical_error": execution_test.get("error")  # For debugging
            }
        
        # Add model info if execution was successful
        test_results = execution_test.get("test_results", {})
        if test_results.get("success", False) and "steps" in test_results:
            model_info = test_results["steps"].get("get_model_info", {})
            if model_info and "inputs" in model_info:
                summary["model_info"] = {
                    "input_count": len(model_info.get("inputs", [])),
                    "output_count": len(model_info.get("outputs", [])),
                    "model_size_mb": model_info.get("model_size_mb", 0),
                    "framework": test_results.get("framework")
                }
        
        return summary
