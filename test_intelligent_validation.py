#!/usr/bin/env python3
"""
Test script for the intelligent AI validation system
This shows how the AI compares descriptions with actual file contents
"""

import requests
import json
import zipfile
import io

def create_image_classification_model():
    """Create a realistic image classification model zip"""
    
    model_code = """
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(ImageClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Model configuration
model = ImageClassifier(num_classes=10)
"""

    config_content = """
{
    "model_type": "image_classification",
    "framework": "pytorch",
    "input_shape": [3, 224, 224],
    "num_classes": 10,
    "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
}
"""

    readme_content = """
# CIFAR-10 Image Classification Model

This model classifies images into 10 categories from the CIFAR-10 dataset.

## Architecture
- Custom CNN with 5 convolutional layers
- 3 fully connected layers
- ReLU activations and dropout for regularization

## Usage
```python
model = torch.load('model.pth')
model.eval()
output = model(input_tensor)
```
"""

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('model.py', model_code)
        zip_file.writestr('config.json', config_content)
        zip_file.writestr('README.md', readme_content)
        zip_file.writestr('model.pth', b'pretend_model_weights_data' * 100)  # Fake weights
        
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_text_generation_model():
    """Create a realistic text generation model zip"""
    
    model_code = """
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGenerator(nn.Module):
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, 12),
            num_layers
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        output = self.transformer(embeddings)
        logits = self.lm_head(output)
        return logits

# Load pre-trained model
model = TextGenerator()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
"""

    config_content = """
{
    "model_type": "text_generation",
    "framework": "pytorch",
    "vocab_size": 50257,
    "max_length": 1024,
    "temperature": 0.7
}
"""

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('model.py', model_code)
        zip_file.writestr('config.json', config_content)
        zip_file.writestr('model.pth', b'pretend_model_weights_data' * 100)
        
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def test_intelligent_validation():
    """Test the intelligent AI validation system"""
    
    print("=== Testing Intelligent AI Validation ===\n")
    
    test_cases = [
        # Test Case 1: Perfect match - should be VALID
        {
            "name": "Perfect Match: Image Classification",
            "model_zip": create_image_classification_model(),
            "description": "CIFAR-10 image classification model with CNN architecture for categorizing images into 10 classes",
            "setup": "Load with torch.load('model.pth') and use model.eval() for inference on 224x224 RGB images",
            "expected": "VALID",
            "reason": "Description perfectly matches the actual CNN model for image classification"
        },
        
        # Test Case 2: Brief but accurate - should be VALID  
        {
            "name": "Brief but Accurate Description",
            "model_zip": create_image_classification_model(),
            "description": "image classifier",
            "setup": "load and run",
            "expected": "VALID",
            "reason": "Even though brief, description matches what the model actually does"
        },
        
        # Test Case 3: Gibberish but AI should check files - should be INVALID
        {
            "name": "Gibberish vs Real Model Files",
            "model_zip": create_image_classification_model(),
            "description": "adsada",
            "setup": "hjkl qwerty",
            "expected": "INVALID",
            "reason": "No meaningful connection between gibberish description and actual model functionality"
        },
        
        # Test Case 4: Wrong task type - should be INVALID
        {
            "name": "Wrong Task Type Claim",
            "model_zip": create_image_classification_model(),
            "description": "This is a text generation model for creating stories and articles",
            "setup": "Use for generating text with GPT-like capabilities",
            "expected": "INVALID", 
            "reason": "Claims text generation but files show image classification model"
        },
        
        # Test Case 5: Correct match for text generation - should be VALID
        {
            "name": "Correct Text Generation Match",
            "model_zip": create_text_generation_model(),
            "description": "GPT-style text generation model for creating coherent text",
            "setup": "Load with transformers library and use for text generation tasks",
            "expected": "VALID",
            "reason": "Description matches the transformer-based text generation model"
        },
        
        # Test Case 6: Very generic but not wrong - might be VALID
        {
            "name": "Generic but Not Wrong",
            "model_zip": create_image_classification_model(),
            "description": "machine learning model",
            "setup": "use for AI tasks",
            "expected": "VALID",  # Could go either way, but should be valid since it's technically correct
            "reason": "Generic but not incorrect - the model IS a machine learning model"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 70)
        print(f"Description: '{test_case['description']}'")
        print(f"Setup: '{test_case['setup']}'")
        print(f"Expected: {test_case['expected']}")
        print(f"Reasoning: {test_case['reason']}")
        print()
        
        files = {'file': (f'test_model_{i}.zip', test_case['model_zip'], 'application/zip')}
        data = {
            'model_name': f'Test Model {i}',
            'model_setUp': test_case['setup'],
            'description': test_case['description']
        }
        
        try:
            response = requests.post('http://localhost:8000/api/models/model-upload', files=files, data=data, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                actual_status = result.get('status')
                reason = result.get('reason', 'N/A')
                
                print(f"🤖 AI Decision: {actual_status}")
                if reason != 'N/A':
                    print(f"   AI Reasoning: {reason}")
                
                # Check if result matches expectation
                if actual_status == test_case['expected']:
                    print(f"✅ CORRECT: AI made the right decision!")
                else:
                    print(f"🤔 DIFFERENT: Expected {test_case['expected']}, AI decided {actual_status}")
                    print(f"   This might actually be a better judgment by the AI")
                    
                # Show task detection
                task_detection = result.get('task_detection', {})
                if task_detection:
                    task_type = task_detection.get('task_type', 'N/A')
                    confidence = task_detection.get('confidence', 'N/A')
                    print(f"   Task Detected: {task_type} (confidence: {confidence})")
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out (AI is thinking hard about this one!)")
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure server is running on http://localhost:8000")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    print("🧠 This test shows how the AI intelligently compares descriptions with actual model files")
    print("💡 The AI will analyze the relationship between claims and reality, not just reject based on rules\n")
    test_intelligent_validation()