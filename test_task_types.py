#!/usr/bin/env python3
"""
Test script to demonstrate the task type mapping system
"""

from app.task_types import normalize_task_type, get_task_type_suggestions, TASK_TYPES

def test_task_type_mapping():
    """Test the task type normalization system"""
    
    print("=== Task Type Mapping System Test ===\n")
    
    # Test cases: AI might detect these variations
    test_cases = [
        "image_classification",
        "Image Classification", 
        "classify images",
        "GPT",
        "large language model",
        "text generation",
        "object detection",
        "YOLO",
        "sentiment analysis",
        "chatbot",
        "translation",
        "OCR",
        "speech to text",
        "diffusion",
        "stable diffusion",
        "unknown_task_type"
    ]
    
    print("Testing AI-detected task types → Standardized task types:")
    print("-" * 60)
    
    for test_input in test_cases:
        normalized = normalize_task_type(test_input)
        print(f"'{test_input}' → '{normalized}'")
    
    print(f"\n=== Available Standard Task Types ({len(TASK_TYPES)}) ===")
    for i, task_type in enumerate(TASK_TYPES, 1):
        print(f"{i:2d}. {task_type}")
    
    print(f"\n=== Task Type Suggestions ===")
    test_queries = ["image", "text", "audio", "classification"]
    
    for query in test_queries:
        suggestions = get_task_type_suggestions(query)
        print(f"Query: '{query}' → {suggestions[:3]}...")  # Show first 3 suggestions

if __name__ == "__main__":
    test_task_type_mapping()