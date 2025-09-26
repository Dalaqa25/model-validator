from typing import Union, Optional

# Standard task types used throughout the system
TASK_TYPES = [
    "image-classification",
    "object-detection",
    "text-generation",
    "translation",
    "sentiment-analysis",
    "speech-to-text",
    "text-to-speech",
    "image-generation",
    "text-classification",
    "named-entity-recognition",
    "question-answering",
    "summarization",
    "language-modeling",
    "feature-extraction",
    "fill-mask",
    "token-classification",
    "zero-shot-classification",
    "conversational",
    "table-question-answering",
    "text2text-generation",
    "automatic-speech-recognition",
    "audio-classification",
    "voice-activity-detection",
    "depth-estimation",
    "image-segmentation",
    "image-to-text",
    "visual-question-answering",
    "document-question-answering",
    "video-classification",
    "reinforcement-learning",
    "robotics",
    "tabular-classification",
    "tabular-regression",
    "time-series-forecasting",
    "graph-ml"
]

# AI-friendly variations mapping to standard task types
TASK_TYPE_MAPPING = {
    # Image tasks
    "image_classification": "image-classification",
    "image classification": "image-classification", 
    "classify images": "image-classification",
    "image categorization": "image-classification",
    "visual classification": "image-classification",
    
    "object_detection": "object-detection",
    "object detection": "object-detection",
    "detect objects": "object-detection",
    "bounding box detection": "object-detection",
    "yolo": "object-detection",
    
    "image_generation": "image-generation",
    "image generation": "image-generation",
    "generate images": "image-generation",
    "text to image": "image-generation",
    "text2image": "image-generation",
    "diffusion": "image-generation",
    "stable diffusion": "image-generation",
    "dall-e": "image-generation",
    "midjourney": "image-generation",
    
    "image_segmentation": "image-segmentation",
    "image segmentation": "image-segmentation",
    "semantic segmentation": "image-segmentation",
    "instance segmentation": "image-segmentation",
    
    "image_to_text": "image-to-text",
    "image to text": "image-to-text",
    "image captioning": "image-to-text",
    "visual captioning": "image-to-text",
    "ocr": "image-to-text",
    
    # Text tasks
    "text_generation": "text-generation",
    "text generation": "text-generation",
    "generate text": "text-generation",
    "language generation": "text-generation",
    "gpt": "text-generation",
    "llm": "text-generation",
    "large language model": "text-generation",
    
    "text_classification": "text-classification",
    "text classification": "text-classification",
    "classify text": "text-classification",
    "document classification": "text-classification",
    
    "sentiment_analysis": "sentiment-analysis",
    "sentiment analysis": "sentiment-analysis",
    "emotion detection": "sentiment-analysis",
    "opinion mining": "sentiment-analysis",
    
    "translation": "translation",
    "translate": "translation",
    "language translation": "translation",
    "machine translation": "translation",
    "mt": "translation",
    
    "question_answering": "question-answering",
    "question answering": "question-answering",
    "qa": "question-answering",
    "answer questions": "question-answering",
    
    "summarization": "summarization",
    "summarize": "summarization",
    "text summarization": "summarization",
    "abstract generation": "summarization",
    
    "named_entity_recognition": "named-entity-recognition",
    "named entity recognition": "named-entity-recognition",
    "ner": "named-entity-recognition",
    "entity extraction": "named-entity-recognition",
    
    "token_classification": "token-classification",
    "token classification": "token-classification",
    "sequence labeling": "token-classification",
    "pos tagging": "token-classification",
    
    # Audio tasks
    "speech_to_text": "speech-to-text",
    "speech to text": "speech-to-text",
    "automatic speech recognition": "automatic-speech-recognition",
    "asr": "automatic-speech-recognition",
    "transcription": "speech-to-text",
    "voice recognition": "speech-to-text",
    
    "text_to_speech": "text-to-speech",
    "text to speech": "text-to-speech",
    "tts": "text-to-speech",
    "voice synthesis": "text-to-speech",
    "speech synthesis": "text-to-speech",
    
    "audio_classification": "audio-classification",
    "audio classification": "audio-classification",
    "sound classification": "audio-classification",
    "music classification": "audio-classification",
    
    # Video tasks
    "video_classification": "video-classification",
    "video classification": "video-classification",
    "action recognition": "video-classification",
    "activity recognition": "video-classification",
    
    # Other tasks
    "conversational": "conversational",
    "chatbot": "conversational",
    "dialogue": "conversational",
    "chat": "conversational",
    
    "feature_extraction": "feature-extraction",
    "feature extraction": "feature-extraction",
    "embedding": "feature-extraction",
    "representation learning": "feature-extraction",
    
    "zero_shot_classification": "zero-shot-classification",
    "zero shot classification": "zero-shot-classification",
    "zero-shot": "zero-shot-classification",
    
    "fill_mask": "fill-mask",
    "fill mask": "fill-mask",
    "masked language modeling": "fill-mask",
    "mlm": "fill-mask",
    
    "tabular_classification": "tabular-classification",
    "tabular classification": "tabular-classification",
    "structured data classification": "tabular-classification",
    
    "tabular_regression": "tabular-regression",
    "tabular regression": "tabular-regression",
    "structured data regression": "tabular-regression",
    
    "time_series_forecasting": "time-series-forecasting",
    "time series forecasting": "time-series-forecasting",
    "forecasting": "time-series-forecasting",
    "prediction": "time-series-forecasting",
    
    "reinforcement_learning": "reinforcement-learning",
    "reinforcement learning": "reinforcement-learning",
    "rl": "reinforcement-learning",
    
    "robotics": "robotics",
    "robot control": "robotics",
    "robotic": "robotics",
    
    "graph_ml": "graph-ml",
    "graph machine learning": "graph-ml",
    "graph neural network": "graph-ml",
    "gnn": "graph-ml",
}


def normalize_task_type(detected_task: str) -> Optional[str]:
    """
    Normalize AI-detected task types to standardized task types.
    
    Args:
        detected_task (str): Task type detected by AI (can be variations)
        
    Returns:
        str | None: Standardized task type from TASK_TYPES, or None if no task provided
    """
    if not detected_task:
        return None
        
    # Convert to lowercase and strip
    normalized_input = detected_task.lower().strip()
    
    # Direct mapping check
    if normalized_input in TASK_TYPE_MAPPING:
        return TASK_TYPE_MAPPING[normalized_input]
    
    # Check if it's already a standard task type
    if normalized_input in TASK_TYPES:
        return normalized_input
        
    # Fuzzy matching for common patterns
    for key, value in TASK_TYPE_MAPPING.items():
        if key in normalized_input or normalized_input in key:
            return value
            
    # Return original if no mapping found
    return detected_task


def get_task_type_suggestions(query: str) -> list:
    """
    Get suggested task types based on a query string.
    
    Args:
        query (str): Search query
        
    Returns:
        list: List of matching task types
    """
    if not query:
        return TASK_TYPES
        
    query = query.lower().strip()
    suggestions = []
    
    # Check direct matches in standard types
    for task_type in TASK_TYPES:
        if query in task_type or task_type in query:
            suggestions.append(task_type)
    
    # Check mapping keys
    for key, value in TASK_TYPE_MAPPING.items():
        if query in key and value not in suggestions:
            suggestions.append(value)
            
    return suggestions if suggestions else TASK_TYPES