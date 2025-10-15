from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app import utils
from app.validator import validate_model_zip, validate_model_with_ai
from app.runners.runner_manager import RunnerManager
from app.template_matcher import TemplateMatcher
import logging
import httpx
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize runner manager and template matcher
runner_manager = RunnerManager()
template_matcher = TemplateMatcher()

@router.post("/model-upload")
async def model_upload(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    model_setUp: str = Form(...),
    description: str = Form(...)
):
    logger.info(f"Model upload started for file: {file.filename}")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Model setUp: {model_setUp}")
    logger.info(f"Description: {description}")
    
    # Check for .zip format
    if not file.filename or not file.filename.lower().endswith(".zip"):
        logger.error("Upload failed: Only .zip files are allowed.")
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")
    
    contents = await file.read()
    
    try:
        # Clean the zip file before validation
        logger.info("Cleaning the uploaded zip file...")
        cleaned_contents = utils.clean_zip_file(contents)
        logger.info("Zip file cleaning successful.")

        # Perform initial validation (file types, extensions)
        logger.info("Performing initial file validation...")
        extracted_files_list = utils.list_zip_contents(cleaned_contents)
        framework_used = validate_model_zip(extracted_files_list)
        logger.info(f"Initial validation successful. Detected framework: {framework_used}")

        # Extract content for AI validation
        logger.info("Extracting file contents for AI validation...")
        file_contents = utils.extract_zip_contents(cleaned_contents)

        # Perform AI validation
        logger.info("Performing AI validation...")
        ai_result = validate_model_with_ai(file_contents, description, model_setUp)
        logger.info(f"AI validation finished. Result: {ai_result['status']}")

        # Initialize execution results as None
        execution_summary = None

        # Only proceed if AI validation passes
        if ai_result['status'] == 'VALID':
            # Perform template matching validation
            task_type = ai_result.get('task_detection', 'unknown')
            logger.info(f"AI validation passed. Checking template availability for task_type={task_type}, framework={framework_used}")
            
            template_validation = template_matcher.validate_template_availability(task_type, framework_used)
            template_success = template_validation['template_validation']['success']
            
            if template_success:
                logger.info("Template validation passed. Proceeding with model execution testing...")
                execution_results = runner_manager.test_model_execution(cleaned_contents, framework_used)
                logger.info(f"Execution testing finished. Success: {execution_results['execution_test']['success']}")
                
                # Get clean execution summary for frontend
                execution_summary = runner_manager.get_execution_summary(execution_results)
                
                # If execution testing also passed, upload to Supabase storage
                if execution_results['execution_test']['success']:
                    template_id = template_validation['template_validation']['template_id']
                    logger.info(f"All validations passed. Uploading model to Supabase storage...")
                    
                    # Pass the original filename to match Next.js pattern
                    storage_result = template_matcher.upload_model_to_storage(
                        cleaned_contents, 
                        model_name, 
                        template_id, 
                        original_filename=file.filename
                    )
                    
                    if storage_result['upload_success']:
                        logger.info(f"Model successfully uploaded to storage: {storage_result['storage_path']}")
                        # Add comprehensive storage info to AI result (matching Next.js pattern)
                        ai_result['storage_upload'] = {
                            "success": True,
                            "storage_path": storage_result['storage_path'],
                            "file_storage_info": storage_result['file_storage_info'],  # Complete file metadata
                            "template_id": template_id
                        }
                    else:
                        logger.error(f"Failed to upload model to storage: {storage_result['user_friendly_message']}")
                        # Update AI result to reflect storage upload failure
                        ai_result['status'] = 'INVALID'
                        ai_result['reason'] = storage_result['user_friendly_message']
                        ai_result['storage_upload'] = {
                            "success": False,
                            "error": storage_result['error'],
                            "user_friendly_message": storage_result['user_friendly_message'],
                            "suggestions": storage_result['suggestions']
                        }
                else:
                    logger.info("Execution testing failed. Skipping storage upload.")
            else:
                logger.info("Template validation failed. Skipping model execution testing.")
                execution_summary = None
                # Update AI result to reflect template validation failure
                ai_result['status'] = 'INVALID'
                ai_result['reason'] = template_validation['template_validation']['user_friendly_message']
                ai_result['template_validation'] = template_validation['template_validation']
            
            # Forward the ZIP file and full JSON response to the other server
            full_response = {
                "status": ai_result.get("status"),
                "reason": ai_result.get("reason"),
                "framework_used": framework_used,
                "task_detection": ai_result.get("task_detection", {}),
                "execution_test": execution_summary,
                "template_validation": ai_result.get("template_validation"),
                "storage_upload": ai_result.get("storage_upload")
            }
            try:
                logger.info("Sending ZIP file and full JSON to 127.0.0.1:8001/upload/")
                async with httpx.AsyncClient() as client:
                    files = {'file': ('model.zip', cleaned_contents, 'application/zip')}
                    data = {'text': json.dumps(full_response)}
                    response = await client.post("http://127.0.0.1:8001/upload/", files=files, data=data)
                    if response.status_code == 200:
                        logger.info("Successfully sent ZIP file and full JSON to 127.0.0.1:upload/")
                    else:
                        logger.error(f"Failed to send ZIP file and full JSON. Status: {response.status_code}, Reason: {response.text}")
            except Exception as e:
                logger.error(f"Error forwarding ZIP file and full JSON: {str(e)}")
        else:
            logger.info("AI validation failed. Skipping model execution testing.")

        # Include detected framework, task type, execution results, template validation, and storage upload in the response
        response = {
            "status": ai_result.get("status"),
            "reason": ai_result.get("reason"),
            "framework_used": framework_used,
            "task_detection": ai_result.get("task_detection", {}),
            "execution_test": execution_summary,
            "template_validation": ai_result.get("template_validation"),
            "storage_upload": ai_result.get("storage_upload")
        }
        # Remove None values
        response = {k: v for k, v in response.items() if v is not None}

        return response

    except HTTPException as e:
        logger.error(f"Validation failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during model processing.")