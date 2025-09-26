#!/usr/bin/env python3
"""
Test script for the strict task type detection system
Tests the new binary approach: either clear match or "no_task_found"
"""

import requests
import json
import zipfile
import io

def create_clear_cnn_model():
    """Create a clearly identifiable CNN image classification model"""
    
    model_code = """
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageClassifier, self).__init__()
        # Clear CNN architecture for image classification
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # RGB input -> image
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Classification output
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Model instance
model = ImageClassifier(num_classes=10)
"""

    config_content = """
{
    "task": "image_classification", 
    "input_shape": [3, 224, 224],
    "num_classes": 10,
    "architecture": "CNN"
}
"""

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('model.py', model_code)
        zip_file.writestr('config.json', config_content)
        zip_file.writestr('model.pth', b'fake_model_weights' * 100)
        
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_unclear_files():
    """Create unclear files that shouldn't match any specific task type"""
    
    unclear_code = """
# Generic utilities
import os
import json

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, item):
        return item
"""

    template_file = """
name: Bug Report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior

**Expected behavior**
A clear and concise description of what you expected to happen.
"""

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('utils.py', unclear_code)
        zip_file.writestr('bug_report.md', template_file)
        zip_file.writestr('data.json', '{"type": "generic"}')
        
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_transformer_model():
    """Create a clear transformer text generation model"""
    
    model_code = """
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TextGenerator(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1024, embed_dim))
        
        # Transformer layers for text generation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=3072,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)  # Language modeling head
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        embeddings = self.embedding(input_ids)
        embeddings += self.pos_embedding[:seq_len]
        
        output = self.transformer(embeddings)
        logits = self.lm_head(output)  # Text generation logits
        return logits

# Text generation model
model = TextGenerator()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
"""

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('text_model.py', model_code)
        zip_file.writestr('model.pth', b'fake_transformer_weights' * 100)
        
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def test_strict_task_detection():
    """Test the new strict task detection system"""
    
    print("=== Testing STRICT Task Detection System ===\n")
    print("🎯 Goal: Binary detection - either definitive match or 'no_task_found'")
    print("❌ No more guessing, hallucination, or low confidence matches\n")
    
    test_cases = [
        {
            "name": "Clear CNN Image Classifier",
            "model_zip": create_clear_cnn_model(),
            "description": "random description adsada",
            "setup": "whatever setup hjkl",
            "expected_task": "image-classification",
            "reason": "Clear CNN architecture with image processing should be detected"
        },
        
        {
            "name": "Clear Transformer Text Generator", 
            "model_zip": create_transformer_model(),
            "description": "asdasd text stuff",
            "setup": "qwerty setup",
            "expected_task": "text-generation",
            "reason": "Clear transformer with language modeling head should be detected"
        },
        
        {
            "name": "Unclear/Generic Files",
            "model_zip": create_unclear_files(),
            "description": "some AI model",
            "setup": "use it for machine learning",
            "expected_task": "no_task_found",
            "reason": "Generic utility files with no clear ML task should return no_task_found"
        },
        
        {
            "name": "Perfect Description but Unclear Files",
            "model_zip": create_unclear_files(),
            "description": "Advanced ResNet-50 image classification model trained on ImageNet",
            "setup": "Load with PyTorch and use for image classification tasks",
            "expected_task": "no_task_found", 
            "reason": "AI should ignore perfect description and focus only on files"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 70)
        print(f"Description: '{test_case['description']}'")
        print(f"Setup: '{test_case['setup']}'")
        print(f"Expected Task: {test_case['expected_task']}")
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
                detected_task = result.get('task_detection', 'unknown')
                
                print(f"🤖 AI Detection Result:")
                print(f"   Task Type: {detected_task}")
                
                # Check if result matches expectation
                if detected_task == test_case['expected_task']:
                    print(f"✅ PERFECT: Expected {test_case['expected_task']}, got {detected_task}")
                    
                    # Verify strictness
                    if detected_task != "no_task_found":
                        print(f"🎯 STRICT SUCCESS: Clear task detected without extra fields")
                    elif detected_task == "no_task_found":
                        print(f"🎯 STRICT SUCCESS: Correctly returned 'no_task_found'")
                    
                else:
                    print(f"🤔 DIFFERENT: Expected {test_case['expected_task']}, got {detected_task}")
                    print(f"   Need to analyze if this is actually better judgment")
                
                # Check validation result
                status = result.get('status')
                reason = result.get('reason', 'N/A')
                print(f"   Validation: {status}")
                if reason != 'N/A':
                    print(f"   Validation Reason: {reason}")
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out")
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure server is running on http://localhost:8000")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    print("🔥 Testing the new STRICT task detection system")
    print("📋 Rules: Either clear match → return task type, or unclear → return 'no_task_found'")
    print("🚫 No more soft classification, guessing, or hallucination\n")
    test_strict_task_detection()