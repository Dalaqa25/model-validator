import httpx
import os

def run_test():
    """
    Sends a POST request to the model validator with a sample model and random data.
    """
    url = "http://localhost:8000/api/models/upload"
    model_path = "simple_model.zip"

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Make sure the file is in the same directory as this script.")
        return

    with open(model_path, "rb") as f:
        files = {"file": ("simple_model.zip", f.read(), "application/zip")}
        data = {
            "model_name": "Simple Tabular Classifier",
            "model_setUp": "Create neural network with linear layer for tabular data classification",
            "description": "A basic neural network model that takes 10 numerical features as input and outputs predictions for 2 classes in tabular classification tasks"
        }

        try:
            with httpx.Client(timeout=300.0) as client:  # 5 minute timeout for large models
                response = client.post(url, files=files, data=data)

            print(f"Status Code: {response.status_code}")
            try:
                print("Response JSON:")
                print(response.json())
            except Exception:
                print("Response Text:")
                print(response.text)

        except httpx.ConnectError as e:
            print(f"Connection Error: {e}")
            print("Please make sure the model validator server is running.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_test()
