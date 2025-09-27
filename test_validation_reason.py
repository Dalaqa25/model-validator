#!/usr/bin/env python3

import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.validator import validate_model_with_ai

def test_valid_model_response():
    """Test that a valid model response includes a reason"""
    # Mock file contents, description, and setup
    file_contents = """
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.fc(x)
    """
    
    description = "A simple PyTorch model for linear regression"
    model_setup = "Load with PyTorch and use for inference"
    
    # Call the validation function
    result = validate_model_with_ai(file_contents, description, model_setup)
    
    print("Validation result:")
    print(f"Status: {result.get('status')}")
    print(f"Reason: {result.get('reason')}")
    print(f"Task detection: {result.get('task_detection')}")
    print(f"Validation status: {result.get('validation_status')}")
    
    # Check if reason is included
    if 'reason' in result:
        print("\n✓ Reason is included in the response")
    else:
        print("\n✗ Reason is missing from the response")

if __name__ == "__main__":
    test_valid_model_response()