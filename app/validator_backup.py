from fastapi import HTTPException
from .framework_extentions import extension_mapping, get_clean_framework_name
from app.open_router import ask_openrouter
from app.task_types import normalize_task_type, TASK_TYPES
import json
import re


def detect_task_type(description: str, model_setup: str, file_contents: str) -> dict:
    """
    Strictly detects task type from model files. Only returns a task type if there's a clear match.
    No guessing, no hallucination - either it matches or it doesn't.
    """
    
    # Import task types for strict matching
    from app.task_types import TASK_TYPES
    
    prompt = f"""
    You are a STRICT task type detector. Your job is to determine if the model files clearly match one of the predefined task types.
    
    ### STRICT RULES:
    1. **ONLY** return a task type if you can clearly identify it from the actual model files
    2. **DO NOT GUESS** - if uncertain, return "no_task_found"
    3. **DO NOT HALLUCINATE** - only use evidence from the actual files
    4. **IGNORE** vague descriptions - focus on file contents only
    5. **BE BINARY** - either it clearly matches or it doesn't
    
    ### Available Task Types:
    {', '.join(TASK_TYPES)}
    
    ### Analysis Process:
    1. Examine the model files (code, architecture, imports)
    2. Look for clear indicators of specific tasks
    3. If you can definitively identify the task → return it
    4. If unclear, uncertain, or no clear match → return "no_task_found"
    
    ### Examples of CLEAR matches:
    - CNN with image inputs → "image-classification" 
    - Transformer with text tokenization → "text-generation"
    - YOLO architecture → "object-detection"
    - Audio processing with speech → "speech-to-text"
    
    ### Examples of NO CLEAR MATCH:
    - Generic neural network without clear purpose
    - Empty or template files
    - Configuration files only
    - Unclear architecture
    
    ### Current Files to Analyze:
    ```
    {file_contents[:2000]}
    ```
    
    ### Your Decision:
    Analyze ONLY the file contents. Ignore the user description completely.
    
    Return JSON: {{"task_type": "specific-task-type"}} or {{"task_type": "no_task_found"}}
    No confidence levels, no reasoning, no uncertainty - just binary detection.
    """

    try:
        ai_response = ask_openrouter(prompt)
        
        # Clean the response
        cleaned_response = ai_response.strip()
        cleaned_response = re.sub(r'^```json\s*', '', cleaned_response, flags=re.MULTILINE)
        cleaned_response = re.sub(r'\s*```$', '', cleaned_response, flags=re.MULTILINE)
        cleaned_response = cleaned_response.strip()
        
        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {
                "task_type": "no_task_found",
                "confidence": "none",
                "reasoning": "AI response parsing failed",
                "error": "parsing_failed"
            }

        # Get the detected task
        detected_task = parsed.get("task_type")
        
        # If no clear task found, return accordingly
        if not detected_task or detected_task == "no_task_found":
            return {
                "task_type": "no_task_found",
                "confidence": "none",
                "reasoning": "No clear task type match found in model files",
                "original_detection": detected_task or "none"
            }
        
        # Normalize the detected task type
        normalized_task = normalize_task_type(detected_task)
        
        # Verify it's actually in our task types list
        if normalized_task not in TASK_TYPES:
            return {
                "task_type": "no_task_found",
                "confidence": "none", 
                "reasoning": "Detected task type not in predefined list",
                "original_detection": detected_task
            }
        
        # Return successful strict detection
        return {
            "task_type": normalized_task,
            "confidence": "definitive",
            "reasoning": f"Clear match found: {normalized_task}",
            "original_detection": detected_task
        }

    except Exception as e:
        return {
            "task_type": "no_task_found",
            "confidence": "none", 
            "reasoning": "Task detection service failed",
            "error": str(e)
        }


def validate_model_zip(extracted_files):
    """
    Uses AI to intelligently validate if the file contents match the provided description and setup.
    The AI analyzes the actual model files and compares them with the user's description.
    """
    
    # Get task detection first for context
    task_detection = detect_task_type(description, model_setup, file_contents)
    
    # Enhanced AI validation prompt - focuses on intelligent comparison
    prompt = f"""
    You are an expert AI model validator. Your job is to intelligently analyze if the user's description and setup instructions match the actual model files they uploaded.
    
    ### Your Analysis Process:
    1. **Examine the actual model files** - Look at file structure, code, architecture, imports, etc.
    2. **Understand the user's intent** - What do they claim this model does? 
    3. **Compare intelligently** - Does their description align with what the files actually contain?
    4. **Consider context** - Even brief descriptions can be valid if they match the files
    
    ### Validation Rules:
    - ✅ **VALID**: If description/setup reasonably matches the file contents (even if brief)
    - ✅ **VALID**: If the user's intent aligns with what the model actually does  
    - ✅ **VALID**: If setup instructions work with the provided files
    - ❌ **INVALID**: Only if there's a clear mismatch between claims and reality
    - ❌ **INVALID**: If description claims functionality not present in files
    - ❌ **INVALID**: If setup instructions won't work with these files
    
    ### Examples:
    - Description: "image classifier" + Files: CNN model → **VALID** (matches)
    - Description: "adsada" + Files: working PyTorch model → **INVALID** (no meaningful connection)
    - Description: "text generator" + Files: image processing code → **INVALID** (mismatch)
    - Description: "model" + Files: complete working model → **VALID** (brief but accurate)
    
    ### Current Analysis:
    
    **Detected Task Type:** {task_detection.get('task_type', 'unknown')} (confidence: {task_detection.get('confidence', 'unknown')})
    
    **User's Description:**
    "{description}"
    
    **User's Setup Instructions:**
    "{model_setup}"
    
    **Actual Model Files Content:**
    ```
    {file_contents[:3000]}
    ```
    
    ### Your Decision:
    Analyze the relationship between what the user claims and what the files actually contain.
    Focus on whether their description makes sense given the actual model files.
    
    Respond with JSON only: {{"status": "VALID"}} or {{"status": "INVALID", "reason": "specific mismatch explanation"}}
    """

    try:
        ai_response = ask_openrouter(prompt)

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
        return parsed

    except Exception as e:
        return {
            "status": "ERROR",
            "reason": "AI validation service temporarily unavailable",
            "task_detection": task_detection
        }


def validate_model_zip(extracted_files):
    # ------ მარტივი ფაილის ვალიდაციები ------
    # მავნე გაფრთოებების შემოწმება
    BAD_MODEL_EXTENSIONS = (".exe", ".bat", ".cmd", ".sh", ".vbs", ".ps1", ".msi", ".tar", ".rar")
    for file in extracted_files:
        if file.endswith(BAD_MODEL_EXTENSIONS):
            raise HTTPException(status_code=400, detail=f"Bad file found: {file}")

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