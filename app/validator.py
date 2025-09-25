from fastapi import HTTPException
from .framework_extentions import extension_mapping, get_clean_framework_name
from app.open_router import ask_openrouter



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
            # განვსაზღვროთ რომელ framework ეკუთვის ფაილი
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
    Uses Mistral Small 3.2 24B Instruct to validate if the file contents match
    the provided description and setup. Forces strict JSON output.
    """
    prompt = f"""
    You are a strict AI model validator. 
    Analyze if the uploaded files match the provided description and setup instructions.

    ### Rules:
    1. If the description/setup are vague, generic, or gibberish (e.g., "good model", "please use it"), mark as INVALID.
    2. If the description/setup are plausible but clearly unrelated to the file contents, mark as INVALID.
    3. If they reasonably match, mark as VALID.
    4. Output must be STRICT JSON only, with no extra text. Allowed formats:
       {{"status": "VALID"}}
       or
       {{"status": "INVALID", "reason": "your explanation"}}

    ### Input
    Description:
    {description}

    Setup Instructions:
    {model_setup}

    File Contents:
    ```
    {file_contents}
    ```

    ### Output
    Respond only with JSON:
    """

    try:
        ai_response = ask_openrouter(prompt)

        # Try to parse JSON response
        import json
        import re
        
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
                "reason": "AI validation failed - invalid response format"
            }

        # Validate required keys
        if "status" not in parsed:
            return {
                "status": "INVALID",
                "reason": "AI validation error - missing status"
            }

        return parsed

    except Exception as e:
        return {
            "status": "ERROR",
            "reason": "AI validation service temporarily unavailable"
        }