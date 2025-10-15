"""
Template matching functionality to validate if a matching template exists in the database.
"""

import logging
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env.local")

logger = logging.getLogger(__name__)

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase not installed. Template matching will not be available.")


class TemplateMatcher:
    """
    Handles template matching against the database to ensure proper template assignment.
    """
    
    def __init__(self):
        self.supabase: Optional[Client] = None
        self.initialized = False
        
        if SUPABASE_AVAILABLE:
            self._initialize_supabase()
    
    def _initialize_supabase(self):
        """Initialize Supabase client."""
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            
            if not supabase_url or not supabase_key:
                logger.error("Supabase credentials not found in environment variables")
                return
            
            self.supabase = create_client(supabase_url, supabase_key)
            self.initialized = True
            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
    
    def find_matching_template(self, task_type: str, framework: str) -> Dict[str, Any]:
        """
        Find a matching template in the database based on task_type and framework.
        
        Args:
            task_type (str): The detected task type (e.g., "image-classification")
            framework (str): The detected framework (e.g., "ONNX", "PyTorch")
            
        Returns:
            Dict[str, Any]: Template matching results
        """
        result = {
            "template_found": False,
            "template_id": None,
            "error": None,
            "user_friendly_message": None,
            "suggestions": []
        }
        
        if not self.initialized or not self.supabase:
            result["error"] = "Database connection not available"
            result["user_friendly_message"] = "Template matching service is temporarily unavailable"
            result["suggestions"] = [
                "Please try again later",
                "Contact support if the issue persists",
                "Check your database configuration"
            ]
            return result
        
        try:
            logger.info(f"Searching for template: task_type={task_type}, framework={framework}")
            
            # Query templates table with the same logic as your database function
            response = self.supabase.table("templates").select("id, task_type, framework").execute()
            
            if not response.data:
                result["error"] = "No templates found in database"
                result["user_friendly_message"] = "No templates are configured in the system"
                result["suggestions"] = [
                    "Contact support to configure templates",
                    "Check if the templates table is properly set up"
                ]
                return result
            
            # Find matching template using the same logic as your database function
            matched_template = None
            
            for template in response.data:
                template_task_type = template.get("task_type")
                template_framework = template.get("framework", [])
                
                # Check if task_type matches
                if template_task_type != task_type:
                    continue
                
                # Check framework match: either empty array (matches any) or contains our framework
                if template_framework == [] or framework in template_framework:
                    matched_template = template
                    break
            
            if matched_template:
                result["template_found"] = True
                result["template_id"] = matched_template["id"]
                logger.info(f"Found matching template: {matched_template['id']}")
            else:
                result["error"] = f"No matching template found for task_type={task_type} and framework={framework}"
                result["user_friendly_message"] = "No template configuration found for this model type"
                result["suggestions"] = [
                    f"Request template configuration for {task_type} with {framework}",
                    "Contact support to add support for this model type",
                    "Check if you're using a supported framework",
                    "Verify the task type is correct"
                ]
                logger.warning(f"No matching template found for {task_type} + {framework}")
                
        except Exception as e:
            logger.error(f"Error during template matching: {str(e)}")
            result["error"] = str(e)
            result["user_friendly_message"] = "Template matching service encountered an error"
            result["suggestions"] = [
                "Please try again later",
                "Contact support with error details",
                "Check your database connection"
            ]
        
        return result
    
    def upload_model_to_storage(self, zip_contents: bytes, model_name: str, template_id: str, original_filename: str = None, task_type: str = None, framework: str = None) -> Dict[str, Any]:
        """
        Upload the validated model to Supabase storage bucket 'models'.
        Follows the same pattern as the Next.js implementation.
        
        Args:
            zip_contents (bytes): The ZIP file contents to upload
            model_name (str): Name of the model for storage
            template_id (str): The template ID that was matched
            original_filename (str): Original filename (e.g., "my-model.zip")
            
        Returns:
            Dict[str, Any]: Upload results with comprehensive file storage info
        """
        result = {
            "upload_success": False,
            "storage_path": None,
            "file_storage_info": None,
            "error": None,
            "user_friendly_message": None,
            "suggestions": []
        }
        
        if not self.initialized or not self.supabase:
            result["error"] = "Database connection not available"
            result["user_friendly_message"] = "Storage service is temporarily unavailable"
            result["suggestions"] = [
                "Please try again later",
                "Contact support if the issue persists"
            ]
            return result
        
        try:
            # Generate filename following Next.js pattern: timestamp-original-filename
            import time
            from datetime import datetime
            
            timestamp = int(time.time() * 1000)  # Use milliseconds like Date.now() in JS
            
            # Use original filename if provided, otherwise use model_name with .zip extension
            if original_filename:
                file_name = original_filename
            else:
                file_name = f"{model_name}.zip" if not model_name.endswith('.zip') else model_name
            
            # Create storage path following Next.js pattern
            storage_path = f"{timestamp}-{file_name}"
            
            logger.info(f"Uploading model to storage: {storage_path}")
            
            # Upload to Supabase storage bucket 'models' with same options as Next.js
            response = self.supabase.storage.from_("models").upload(
                path=storage_path,
                file=zip_contents,
                file_options={
                    "content-type": "application/zip",
                    "cacheControl": "3600",  # Match Next.js parameter name
                    "upsert": False  # Don't overwrite existing files
                }
            )
            
            if response:
                result["upload_success"] = True
                result["storage_path"] = storage_path
                
                # Create file storage info object matching Next.js pattern
                file_storage_info = {
                    "type": "zip",
                    "fileName": file_name,                    # Original filename
                    "fileSize": len(zip_contents),            # File size in bytes
                    "mimeType": "application/zip",            # MIME type
                    "supabasePath": storage_path,             # Path in Supabase storage
                    "uploadedAt": datetime.now().isoformat() + "Z"  # Upload timestamp (ISO format)
                }
                
                result["file_storage_info"] = file_storage_info
                logger.info(f"Successfully uploaded model to storage: {storage_path}")

                # Persist file storage info into 'models' table
                try:
                    db_payload = {
                        "file_storage": file_storage_info,
                        "model_name": model_name,
                        "template_id": template_id,
                        # Provide fields required by assign_template_id trigger
                        "task_type": task_type,
                        "framework": framework,
                    }
                    db_response = self.supabase.table("models").insert(db_payload).execute()
                    # Supabase python client returns an object with .data
                    if getattr(db_response, "data", None) and len(db_response.data) > 0:
                        created = db_response.data[0]
                        result["db_saved"] = True
                        result["model_id"] = created.get("id")
                    else:
                        result["db_saved"] = False
                        result["db_error"] = "Empty response from database insert"
                except Exception as db_e:
                    logger.error(f"Failed to save file_storage into models table: {str(db_e)}")
                    result["db_saved"] = False
                    result["db_error"] = str(db_e)
            else:
                result["error"] = "Upload response was empty"
                result["user_friendly_message"] = "Failed to upload model to storage"
                result["suggestions"] = [
                    "Please try again",
                    "Contact support if the issue persists"
                ]
                
        except Exception as e:
            logger.error(f"Error uploading model to storage: {str(e)}")
            result["error"] = str(e)
            
            # Categorize storage errors
            error_str = str(e).lower()
            if "duplicate" in error_str or "already exists" in error_str:
                result["user_friendly_message"] = "A model with this name already exists"
                result["suggestions"] = [
                    "Try uploading with a different name",
                    "Contact support if you need to replace an existing model"
                ]
            elif "permission" in error_str or "unauthorized" in error_str:
                result["user_friendly_message"] = "Permission denied for storage upload"
                result["suggestions"] = [
                    "Contact support to check your permissions",
                    "Verify storage bucket configuration"
                ]
            elif "quota" in error_str or "limit" in error_str:
                result["user_friendly_message"] = "Storage quota exceeded"
                result["suggestions"] = [
                    "Contact support to increase storage limits",
                    "Try uploading a smaller model"
                ]
            else:
                result["user_friendly_message"] = "Failed to upload model to storage"
                result["suggestions"] = [
                    "Please try again later",
                    "Contact support with error details"
                ]
        
        return result

    def validate_template_availability(self, task_type: str, framework: str) -> Dict[str, Any]:
        """
        Validate if a template is available for the given task_type and framework.
        This is the main validation function to be called from the API.
        
        Args:
            task_type (str): The detected task type
            framework (str): The detected framework
            
        Returns:
            Dict[str, Any]: Validation results with user-friendly messages
        """
        validation_result = {
            "template_validation": {
                "success": False,
                "template_found": False,
                "template_id": None,
                "error_category": None,
                "user_friendly_message": None,
                "suggestions": []
            }
        }
        
        # Get template matching results
        template_result = self.find_matching_template(task_type, framework)
        
        validation_result["template_validation"]["template_found"] = template_result["template_found"]
        validation_result["template_validation"]["template_id"] = template_result["template_id"]
        
        if template_result["template_found"]:
            validation_result["template_validation"]["success"] = True
            validation_result["template_validation"]["user_friendly_message"] = "Template configuration found"
        else:
            # Categorize the error
            error = template_result.get("error", "Unknown error")
            
            if "connection" in error.lower() or "database" in error.lower():
                validation_result["template_validation"]["error_category"] = "database_connection"
            elif "no template" in error.lower() or "no matching" in error.lower():
                validation_result["template_validation"]["error_category"] = "template_not_found"
            elif "not available" in error.lower():
                validation_result["template_validation"]["error_category"] = "service_unavailable"
            else:
                validation_result["template_validation"]["error_category"] = "unknown"
            
            validation_result["template_validation"]["user_friendly_message"] = template_result["user_friendly_message"]
            validation_result["template_validation"]["suggestions"] = template_result["suggestions"]
        
        return validation_result
