import zipfile, io

def list_zip_contents(file_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_ref:
        return zip_ref.namelist()

def extract_zip_contents(file_bytes: bytes, max_total_size: int = 100000) -> str:
    """
    Extracts relevant files from a zip, but keeps it concise and structured.
    """
    full_content = ""
    total_size = 0
    
    priority_files = ('.md', '.json', '.yaml', '.yml', '.ini', '.cfg', '.toml')
    code_extensions = ('.py',)
    
    with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir() or file_info.filename.startswith(('__', '.')):
                continue

            lower_name = file_info.filename.lower()

            # Choose which files to include
            if lower_name.endswith(priority_files) or lower_name.endswith(code_extensions):
                try:
                    with zip_ref.open(file_info.filename) as file:
                        content = file.read().decode('utf-8', errors='ignore')

                        # For code files, only take the first 50 lines
                        if lower_name.endswith(code_extensions):
                            content = "\n".join(content.splitlines()[:50])

                        # Truncate if over budget
                        if total_size + len(content) > max_total_size:
                            content = content[:max_total_size - total_size]
                            total_size = max_total_size
                        else:
                            total_size += len(content)

                        full_content += f"\nFILE: {file_info.filename}\nCONTENT:\n{content}\n"

                        if total_size >= max_total_size:
                            break
                except Exception as e:
                    print(f"Could not read file {file_info.filename}: {e}")

    return full_content