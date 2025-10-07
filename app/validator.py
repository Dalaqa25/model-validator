from fastapi import HTTPException
from .framework_extentions import extension_mapping, get_clean_framework_name
from app.open_router import ask_openrouter
from app.task_types import normalize_task_type, TASK_TYPES
import json
import re


def detect_task_type(description: str, model_setup: str, file_contents: str) -> str:
    """
    Strictly detects task type from model files. Only returns a task type if there's a clear match.
    No guessing, no hallucination - either it matches or it doesn't.
    """
    
    prompt = f"""
    Analyze the model code and return the task type from: {', '.join(TASK_TYPES)}

    Code:
    {file_contents[:1500]}

    Return JSON: {{"task_type": "task-type"}} or {{"task_type": "no_task_found"}}
    """

    try:
        ai_response = ask_openrouter(prompt)
        print(f"DEBUG: AI response for task detection: {ai_response}")

        # Clean the response
        cleaned_response = ai_response.strip()
        cleaned_response = re.sub(r'^```json\s*', '', cleaned_response, flags=re.MULTILINE)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response, flags=re.MULTILINE)
        cleaned_response = cleaned_response.strip()
        
        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError:
            return "no_task_found"

        # Get the detected task
        detected_task = parsed.get("task_type")
        
        # If no clear task found, return accordingly
        if not detected_task or detected_task == "no_task_found":
            return "no_task_found"
        
        # Normalize the detected task type
        normalized_task = normalize_task_type(detected_task)
        
        # Verify it's actually in our task types list
        if normalized_task not in TASK_TYPES:
            return "no_task_found"
        
        # Return successful strict detection - just the task type
        return normalized_task

    except Exception as e:
        return "no_task_found"


def validate_model_zip(extracted_files):
    # File cleaning is now handled in routes.py

    # შევამოწმოთ ფაილი შეიცავს თუ არა მოდელის ფორმატებს
    KNOWN_MODEL_EXTENSIONS = tuple(extension_mapping.keys())
    KNOWN_MODEL_FILES = ("pytorch_model.bin", "saved_model.pb")
    model_files_found = []
    model_frameworks = {}

    detected_frameworks = set()  # Use set to avoid duplicates
    
    for file in extracted_files:
        if file.endswith(KNOWN_MODEL_EXTENSIONS):
            model_files_found.append(file)
            # განვსაზღვროთ რომელ framework ეკუთვნის ფაილი
            for ext, framework in extension_mapping.items():
                if file.lower().endswith(ext):
                    detected_frameworks.add(framework)
                    break

    if not model_files_found:
        message = f"No model files found in the zip file."
        try:
            ai_suggestion = ask_openrouter(message)
            if ai_suggestion:
                detail = ai_suggestion
            else:
                detail = "No model files found and the AI assistant did not provide a suggestion."
        except Exception as e:
            detail = f"No model files found and AI check failed: {str(e)}"
        raise HTTPException(status_code=400, detail=detail)

    # Return single framework name or handle multiple frameworks
    if len(detected_frameworks) == 0:
        return None
    
    # Clean up framework names
    cleaned_frameworks = get_clean_framework_name(detected_frameworks)
    
    if len(cleaned_frameworks) == 1:
        return list(cleaned_frameworks)[0]
    else:
        # If multiple frameworks detected, return the primary one or combined name
        frameworks_list = sorted(list(cleaned_frameworks))  # Sort for consistency
        if len(frameworks_list) <= 2:
            return " + ".join(frameworks_list)
        else:
            return f"Multi-framework ({len(frameworks_list)} types)"


def validate_model_with_ai(file_contents: str, description: str, model_setup: str):
    """
    Uses AI to intelligently validate if the file contents match the provided description and setup.
    The AI analyzes the actual model files and compares them with the user's description.
    """
    
    # Get task detection first for context
    task_detection = detect_task_type(description, model_setup, file_contents)
    
    prompt = f"""
    Validate if the description matches the model files.

    Task: {task_detection}
    Description: {description}
    Setup: {model_setup}

    Files:
    {file_contents[:2000]}

    Return JSON: {{"status": "VALID", "reason": "explanation"}} or {{"status": "INVALID", "reason": "explanation"}}
    """

    try:
        ai_response = ask_openrouter(prompt)
        print(f"DEBUG: AI response for validation: {ai_response}")

        # Try to parse JSON response

        # Clean the response - remove markdown code blocks if present
        cleaned_response = ai_response.strip()
        # Remove ```json and ``` if present
        cleaned_response = re.sub(r'^```json\s*', '', cleaned_response, flags=re.MULTILINE)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response, flags=re.MULTILINE)
        cleaned_response = cleaned_response.strip()
        
        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {
                "status": "INVALID",
                "reason": "AI validation failed - invalid response format",
                "task_detection": task_detection
            }

        # Validate required keys
        if "status" not in parsed:
            return {
                "status": "INVALID",
                "reason": "AI validation error - missing status",
                "task_detection": task_detection
            }

        # Include task detection results
        parsed["task_detection"] = task_detection
        # Ensure validation_status is included for consistency
        parsed["validation_status"] = parsed.get("status")
        
        # Ensure a reason is always included
        if "reason" not in parsed:
            if parsed.get("status") == "VALID":
                parsed["reason"] = "Model validation successful. The description and setup instructions match the uploaded model files."
            else:
                parsed["reason"] = "Validation status unclear."
        
        return parsed

    except Exception as e:
        return {
            "status": "ERROR",
            "reason": "AI validation service temporarily unavailable",
            "task_detection": task_detection
        }