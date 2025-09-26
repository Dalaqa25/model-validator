#!/usr/bin/env python3
"""
Example script showing how the task type mapping system works in practice
"""

import requests
import json
import zipfile
import io

def create_sample_model_zip():
    """Create a sample model zip file for testing"""
    
    # Create a simple model file content
    model_content = """
# PyTorch Image Classification Model
import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*26*26, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Save model
model = ImageClassifier()
torch.save(model.state_dict(), 'model.pth')
"""

    config_content = """
{
    "model_type": "image_classification",
    "framework": "pytorch", 
    "input_size": [3, 224, 224],
    "num_classes": 10,
    "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
}
"""

    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('model.py', model_content)
        zip_file.writestr('config.json', config_content)
        zip_file.writestr('model.pth', b'fake_model_weights_data')  # Fake model weights
        
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def test_api_with_task_detection():
    """Test the API with the new task detection system"""
    
    print("=== Testing modelValidator API with Task Detection ===\n")
    
    # Create sample model zip
    model_zip = create_sample_model_zip()
    
    # Test case 1: Image Classification (should detect correctly)
    print("Test Case 1: Image Classification Model")
    print("-" * 40)
    
    files = {
        'file': ('image_classifier.zip', model_zip, 'application/zip')
    }
    
    data = {
        'model_name': 'CIFAR-10 Image Classifier',
        'model_setUp': 'Load with PyTorch and use for classifying CIFAR-10 images. Input size is 3x224x224.',
        'description': 'This model performs image classification on CIFAR-10 dataset with 10 classes including airplanes, cars, animals, etc.'
    }
    
    try:
        response = requests.post('http://localhost:8000/api/models/model-upload', files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response:")
            print(json.dumps(result, indent=2))
            
            # Check task detection
            task_detection = result.get('task_detection', {})
            detected_task = task_detection.get('task_type')
            confidence = task_detection.get('confidence')
            
            print(f"\n📊 Task Detection Results:")
            print(f"   Detected Task: {detected_task}")
            print(f"   Confidence: {confidence}")
            print(f"   Expected: image-classification")
            print(f"   Match: {'✅' if detected_task == 'image-classification' else '❌'}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_different_task_descriptions():
    """Test different ways of describing the same task"""
    
    print("\n=== Testing Different Task Descriptions ===\n")
    
    # Different ways to describe image classification
    test_cases = [
        {
            'description': 'This model does image_classification on photos',
            'setup': 'Use for classify images into categories',
            'expected': 'image-classification'
        },
        {
            'description': 'GPT-style text generation model for creating stories',
            'setup': 'Load with transformers and generate text',
            'expected': 'text-generation'
        },
        {
            'description': 'YOLO object detection model',
            'setup': 'Detect objects and bounding boxes in images',
            'expected': 'object-detection'
        }
    ]
    
    model_zip = create_sample_model_zip()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['expected']}")
        print("-" * 40)
        
        files = {'file': (f'test_model_{i}.zip', model_zip, 'application/zip')}
        data = {
            'model_name': f'Test Model {i}',
            'model_setUp': test_case['setup'],
            'description': test_case['description']
        }
        
        try:
            response = requests.post('http://localhost:8000/api/models/model-upload', files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                task_detection = result.get('task_detection', {})
                detected_task = task_detection.get('task_type')
                confidence = task_detection.get('confidence')
                
                print(f"   Input: '{test_case['description'][:50]}...'")
                print(f"   Detected: {detected_task}")
                print(f"   Expected: {test_case['expected']}")
                print(f"   Confidence: {confidence}")
                print(f"   Match: {'✅' if detected_task == test_case['expected'] else '❌'}")
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print()

if __name__ == "__main__":
    test_api_with_task_detection()
    test_different_task_descriptions()