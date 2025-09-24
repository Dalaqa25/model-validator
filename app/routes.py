from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app import utils
from app.validator import validate_model_zip, validate_model_with_ai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

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
    if not file.filename.lower().endswith(".zip"):
        logger.error("Upload failed: Only .zip files are allowed.")
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")
    
    contents = await file.read()

    try:
        # Perform initial validation (file types, extensions)
        logger.info("Performing initial file validation...")
        extracted_files_list = utils.list_zip_contents(contents)
        validate_model_zip(extracted_files_list)
        logger.info("Initial validation successful.")

        # Extract content for AI validation
        logger.info("Extracting file contents for AI validation...")
        file_contents = utils.extract_zip_contents(contents)

        # Perform AI validation
        logger.info("Performing AI validation...")
        ai_result = validate_model_with_ai(file_contents, description, model_setUp)
        logger.info(f"AI validation finished. Result: {ai_result['status']}")

        return ai_result

    except HTTPException as e:
        logger.error(f"Validation failed: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during model processing.")