
extension_mapping = {
    # PyTorch
    '.pt': 'PyTorch',
    '.pth': 'PyTorch',
    
    # TensorFlow
    '.pb': 'TensorFlow (Frozen Graph)',
    '.h5': 'TensorFlow/Keras (HDF5)',
    '.tflite': 'TensorFlow Lite',
    '.tfjs': 'TensorFlow.js',
    
    # ONNX
    '.onnx': 'ONNX (Cross-platform)',
    
    # Scikit-learn
    '.pkl': 'Scikit-learn/Python Pickle',
    '.pickle': 'Scikit-learn/Python Pickle',
    '.joblib': 'Scikit-learn/Joblib',
    
    # JAX/Flax
    '.msgpack': 'JAX/Flax (MessagePack)',
    
    # Apple Core ML
    '.mlmodel': 'Apple Core ML',
    
    # Caffe
    '.caffemodel': 'Caffe',
    '.prototxt': 'Caffe (Architecture)',
    
    # MXNet
    '.params': 'MXNet (Parameters)',
    '.json': 'MXNet (Symbol) / General JSON',
    
    # PaddlePaddle
    '.pdparams': 'PaddlePaddle (Parameters)',
    '.pdmodel': 'PaddlePaddle (Model)',
    
    '.safetensors': 'SafeTensors (Hugging Face)',
    '.bin': 'PyTorch/Hugging Face Binary',
    '.ggml': 'GGML (llama.cpp)',
    '.gguf': 'GGUF (llama.cpp)',
    '.q4_0': 'GGML Quantized (llama.cpp)',
    '.q4_1': 'GGML Quantized (llama.cpp)',
    '.q8_0': 'GGML Quantized (llama.cpp)',
    '.xml': 'OpenVINO (Intel)',
    '.engine': 'TensorRT (NVIDIA)',
}


def get_clean_framework_name(frameworks_set):
    """Convert verbose framework names to clean, frontend-friendly names"""
    framework_mapping = {
        'PyTorch': 'PyTorch',
        'TensorFlow (Frozen Graph)': 'TensorFlow',
        'TensorFlow/Keras (HDF5)': 'TensorFlow',
        'TensorFlow Lite': 'TensorFlow Lite',
        'TensorFlow.js': 'TensorFlow.js',
        'ONNX (Cross-platform)': 'ONNX',
        'Scikit-learn/Python Pickle': 'Scikit-learn',
        'Scikit-learn/Joblib': 'Scikit-learn',
        'JAX/Flax (MessagePack)': 'JAX/Flax',
        'Apple Core ML': 'Core ML',
        'Caffe': 'Caffe',
        'Caffe (Architecture)': 'Caffe',
        'MXNet (Parameters)': 'MXNet',
        'MXNet (Symbol) / General JSON': 'MXNet',
        'PaddlePaddle (Parameters)': 'PaddlePaddle',
        'PaddlePaddle (Model)': 'PaddlePaddle',
        'SafeTensors (Hugging Face)': 'Hugging Face',
        'PyTorch/Hugging Face Binary': 'PyTorch',
        'GGML (llama.cpp)': 'GGML',
        'GGUF (llama.cpp)': 'GGUF',
        'GGML Quantized (llama.cpp)': 'GGML',
        'OpenVINO (Intel)': 'OpenVINO',
        'TensorRT (NVIDIA)': 'TensorRT'
    }
    
    cleaned_frameworks = set()
    for framework in frameworks_set:
        cleaned_name = framework_mapping.get(framework, framework)
        cleaned_frameworks.add(cleaned_name)
    
    return cleaned_frameworks
