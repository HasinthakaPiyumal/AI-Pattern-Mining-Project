"""Simulated API Knowledge Base for MediMatch AI."""

API_DATABASE = [
    {
        "id": "api_lung_nodule_v1",
        "name": "Lung Nodule Detector Pro",
        "description": "Advanced AI for detecting lung nodules in chest X-rays with high precision.",
        "task": "lung nodule detection",
        "accuracy": 0.98,  # ImageNet accuracy (simulated for medical context)
        "latency_ms": 1500, # milliseconds
        "cost_per_image": 0.10, # USD
        "gpu_memory_gb": 12, # GB
        "interpretability": True, # XAI features available
        "provider": "AI Health Solutions"
    },
    {
        "id": "api_lung_nodule_lite",
        "name": "Lung Nodule Detector Lite",
        "description": "Fast and efficient model for lung nodule screening. Good for quick analysis.",
        "task": "lung nodule detection",
        "accuracy": 0.92,
        "latency_ms": 400,
        "cost_per_image": 0.03,
        "gpu_memory_gb": 6,
        "interpretability": False,
        "provider": "MedAI Innovations"
    },
    {
        "id": "api_bone_fracture_v2",
        "name": "Bone Fracture Analyzer",
        "description": "Detects various types of bone fractures in X-ray images.",
        "task": "bone fracture detection",
        "accuracy": 0.96,
        "latency_ms": 1000,
        "cost_per_image": 0.08,
        "gpu_memory_gb": 10,
        "interpretability": True,
        "provider": "OrthoAI Tech"
    },
    {
        "id": "api_cardiac_echo_segmentation",
        "name": "Cardiac Echo Segmenter",
        "description": "Segments cardiac structures from echocardiogram videos.",
        "task": "cardiac segmentation",
        "accuracy": 0.95,
        "latency_ms": 3000,
        "cost_per_image": 0.15,
        "gpu_memory_gb": 16,
        "interpretability": False,
        "provider": "CardioInsights"
    },
    {
        "id": "api_lung_nodule_ultra",
        "name": "Lung Nodule Detector Ultra",
        "description": "State-of-the-art accuracy for lung nodule detection, but resource intensive.",
        "task": "lung nodule detection",
        "accuracy": 0.995,
        "latency_ms": 2500,
        "cost_per_image": 0.25,
        "gpu_memory_gb": 24,
        "interpretability": True,
        "provider": "Global AI Med"
    },
    {
        "id": "api_general_classifier_v1",
        "name": "General Medical Image Classifier",
        "description": "A versatile classifier for various medical image types. Not task-specific.",
        "task": "general classification",
        "accuracy": 0.85,
        "latency_ms": 200,
        "cost_per_image": 0.01,
        "gpu_memory_gb": 4,
        "interpretability": False,
        "provider": "OpenMed AI"
    }
]
