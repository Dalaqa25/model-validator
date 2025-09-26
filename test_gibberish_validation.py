#!/usr/bin/env python3
"""
Test script for the enhanced validation system that detects gibberish input
"""

import requests
import json
import zipfile
import io

def create_sample_model_zip():
    """Create a sample model zip file for testing"""
    
    model_content = """
# PyTorch Model
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)
"""

    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('model.py', model_content)
        zip_file.writestr('model.pth', b'fake_model_weights')
        
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def test_validation_cases():
    """Test various validation scenarios including gibberish detection"""
    
    print("=== Testing Enhanced Validation System ===\n")
    
    model_zip = create_sample_model_zip()
    
    test_cases = [
        {
            "name": "Gibberish Description (should be INVALID)",
            "description": "adsada",
            "setup": "good setup instructions here",
            "expected_status": "INVALID"
        },
        {
            "name": "Gibberish Setup (should be INVALID)", 
            "description": "This is a proper image classification model",
            "setup": "asdasd qwerty",
            "expected_status": "INVALID"
        },
        {
            "name": "Both Gibberish (should be INVALID)",
            "description": "adsada",
            "setup": "hjkl asdf",
            "expected_status": "INVALID"
        },
        {
            "name": "Too Vague (should be INVALID)",
            "description": "good model",
            "setup": "please use it",
            "expected_status": "INVALID"
        },
        {
            "name": "Random Characters (should be INVALID)",
            "description": "aaaaaa bbbbbb cccccc",
            "setup": "123456 qwerty",
            "expected_status": "INVALID"
        },
        {
            "name": "Proper Description (should be VALID)",
            "description": "PyTorch linear regression model trained on housing data to predict prices based on 10 features including size, location, and age",
            "setup": "Load with torch.load() and use model.eval() for inference. Input should be tensor of shape (batch_size, 10)",
            "expected_status": "VALID"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 60)
        print(f"Description: '{test_case['description']}'")
        print(f"Setup: '{test_case['setup']}'")
        print(f"Expected: {test_case['expected_status']}")
        
        files = {'file': (f'test_model_{i}.zip', model_zip, 'application/zip')}
        data = {
            'model_name': f'Test Model {i}',
            'model_setUp': test_case['setup'],
            'description': test_case['description']
        }
        
        try:
            response = requests.post('http://localhost:8000/api/models/model-upload', files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                actual_status = result.get('status')
                reason = result.get('reason', 'N/A')
                
                print(f"✅ Response Status: {actual_status}")
                if reason != 'N/A':
                    print(f"   Reason: {reason}")
                
                # Check if result matches expectation
                if actual_status == test_case['expected_status']:
                    print(f"🎯 CORRECT: Expected {test_case['expected_status']}, got {actual_status}")
                else:
                    print(f"❌ WRONG: Expected {test_case['expected_status']}, got {actual_status}")
                    
                # Show task detection if available
                task_detection = result.get('task_detection', {})
                if task_detection:
                    print(f"   Task Detection: {task_detection.get('task_type', 'N/A')} (confidence: {task_detection.get('confidence', 'N/A')})")
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out (AI processing might be slow)")
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure server is running on http://localhost:8000")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "="*80 + "\n")

def test_gibberish_detection_locally():
    """Test the gibberish detection function locally"""
    
    print("=== Testing Gibberish Detection Function ===\n")
    
    # Import the function locally
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from app.validator import is_gibberish_text
        
        test_inputs = [
            "adsada",
            "asdasd qwerty", 
            "good model",
            "aaaaaa",
            "hjkl",
            "123456",
            "This is a proper PyTorch model for image classification",
            "test",
            "   ",
            "ab",
            "random text here",
            "lorem ipsum"
        ]
        
        for text in test_inputs:
            result = is_gibberish_text(text)
            status = "🚫 GIBBERISH" if result["is_gibberish"] else "✅ VALID"
            print(f"{status} | '{text}' → {result['reason']}")
            
    except ImportError as e:
        print(f"Cannot import function: {e}")
        print("This test requires the server modules to be importable")

if __name__ == "__main__":
    test_gibberish_detection_locally()
    print("\n" + "="*80 + "\n")
    test_validation_cases()