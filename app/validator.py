from fastapi import HTTPException
from .framework_extentions import extension_mapping
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

    for file in extracted_files:
        if file.endswith(KNOWN_MODEL_EXTENSIONS):
            model_files_found.append(file)
            # განვსაზღვროთ რომელ framework ეკუთვის ფაილი
            for ext, framework in extension_mapping.items():
                if file.lower().endswith(ext):
                    model_frameworks[file] = framework
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

    return model_frameworks


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
        try:
            parsed = json.loads(ai_response.strip())
        except json.JSONDecodeError:
            return {
                "status": "INVALID",
                "reason": f"AI did not return valid JSON: {ai_response}"
            }

        # Validate required keys
        if "status" not in parsed:
            return {
                "status": "INVALID",
                "reason": f"Missing 'status' in AI response: {parsed}"
            }

        return parsed

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during AI validation: {str(e)}"
        )