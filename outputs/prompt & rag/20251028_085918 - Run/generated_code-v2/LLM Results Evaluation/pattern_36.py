import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Any

class FeatureExtractor:
    def __init__(self, model_name="resnet50", device="cpu"):
        self.device = torch.device(device)
        self.model = getattr(models, model_name)(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])) # Remove final classification layer
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze().cpu().numpy()

class ExemplarPoolManager:
    def __init__(self):
        self.unlabeled_image_paths: List[str] = []
        self.unlabeled_features: List[np.ndarray] = []
        self.labeled_data: List[Tuple[str, np.ndarray, Any]] = [] # (image_path, features, label)

    def add_unlabeled_images(self, image_paths: List[str], features: List[np.ndarray]):
        self.unlabeled_image_paths.extend(image_paths)
        self.unlabeled_features.extend(features)

    def propose_for_annotation(self, num_proposals: int = 10, random_state: int = None) -> List[str]:
        if not self.unlabeled_features:
            return []

        features_array = np.array(self.unlabeled_features)
        if features_array.shape[0] <= num_proposals:
            proposed_indices = list(range(features_array.shape[0]))
        else:
            kmeans = KMeans(n_clusters=num_proposals, random_state=random_state, n_init=10)
            kmeans.fit(features_array)
            # Select the images closest to the cluster centroids
            proposed_indices = []
            for i in range(num_proposals):
                distances = np.linalg.norm(features_array - kmeans.cluster_centers_[i], axis=1)
                closest_index = np.argmin(distances)
                if closest_index not in proposed_indices:
                    proposed_indices.append(closest_index)
                if len(proposed_indices) == num_proposals: # Stop if we have enough
                    break

        proposed_image_paths = [self.unlabeled_image_paths[i] for i in proposed_indices]
        return proposed_image_paths

    def add_labeled_exemplars(self, annotated_data: List[Tuple[str, Any]]): # (image_path, label)
        newly_labeled_paths = set()
        for path, label in annotated_data:
            try:
                idx = self.unlabeled_image_paths.index(path)
                features = self.unlabeled_features[idx]
                self.labeled_data.append((path, features, label))
                newly_labeled_paths.add(path)
            except ValueError:
                print(f"Warning: Image path {path} not found in unlabeled pool.")
        
        # Remove newly labeled items from unlabeled pool
        self.unlabeled_image_paths = [path for path in self.unlabeled_image_paths if path not in newly_labeled_paths]
        # Re-extract features for the remaining unlabeled paths, or re-index features if too complex
        # For simplicity, we'll assume features are always re-aligned with paths. 
        # A more robust solution might use dictionaries or proper indexing.
        # This simplistic removal means features need to be re-computed or carefully managed.
        # For this example, we'll re-align features based on remaining paths.
        remaining_features = []
        original_path_to_feature = dict(zip(self.unlabeled_image_paths, self.unlabeled_features))
        self.unlabeled_features = [original_path_to_feature[path] for path in self.unlabeled_image_paths]


    def get_labeled_exemplars(self) -> List[Tuple[str, np.ndarray, Any]]:
        return self.labeled_data

    def get_unlabeled_pool_size(self) -> int:
        return len(self.unlabeled_image_paths)

# Example Usage (demonstrates the flow, not part of the core classes)
if __name__ == "__main__":
    # Setup
    extractor = FeatureExtractor(device="cpu") # Use "cuda" if a GPU is available
    manager = ExemplarPoolManager()

    # Simulate a large pool of unlabeled images
    # In a real scenario, these would be actual image files.
    # We'll create dummy paths and features for demonstration.
    dummy_unlabeled_image_paths = [f"path/to/unlabeled_image_{i}.jpg" for i in range(100)]
    dummy_unlabeled_features = [np.random.rand(2048) for _ in range(100)] # Example ResNet50 feature size

    print(f"Initial unlabeled pool size: {manager.get_unlabeled_pool_size()}")
    manager.add_unlabeled_images(dummy_unlabeled_image_paths, dummy_unlabeled_features)
    print(f"Unlabeled pool size after adding initial images: {manager.get_unlabeled_pool_size()}")

    # Stage 1: Model proposes useful unlabeled candidates for human annotation
    num_proposals = 5
    proposed_images = manager.propose_for_annotation(num_proposals=num_proposals, random_state=42)
    print(f"Proposed {len(proposed_images)} images for annotation: {proposed_images}")

    # Simulate human annotation (radiologist provides labels)
    simulated_annotations = [
        (proposed_images[0], "RareDiseaseA"),
        (proposed_images[1], "CommonConditionB"),
        (proposed_images[2], "RareDiseaseC"),
        (proposed_images[3], "CommonConditionA"),
        (proposed_images[4], "RareDiseaseA"),
    ]

    # Add the newly labeled exemplars to the labeled pool
    manager.add_labeled_exemplars(simulated_annotations)
    print(f"Unlabeled pool size after annotation: {manager.get_unlabeled_pool_size()}")
    print(f"Labeled exemplars count: {len(manager.get_labeled_exemplars())}")

    # Stage 2: Use the labeled pool for FewShot Prompting
    current_labeled_data = manager.get_labeled_exemplars()
    print("\n--- Current Labeled Data for FewShot Prompting ---")
    for path, features, label in current_labeled_data:
        print(f"Image: {path}, Label: {label}, Features shape: {features.shape}")

    # Continue the cycle (e.g., propose more, add more labels, update few-shot model)
    # For a real application, the few-shot prompting model would integrate here, 
    # taking `current_labeled_data` as its exemplars.
