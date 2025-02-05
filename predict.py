# import os
# import torch
# import random
# import logging
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from datasets import load_dataset
# from torchvision import transforms
# from typing import List, Tuple, Dict

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class TestingPipeline:
#     def __init__(
#         self,
#         checkpoint_path: str,
#         device: str = 'mps',
#         threshold: float = 0.2
#     ):
#         self.device = device if torch.backends.mps.is_available() else 'cpu'
#         self.threshold = threshold
#         self.classes = ['form', 'letter', 'invoice', 'budget', 'specification']
        
#         # Load the test dataset
#         self.dataset = load_dataset("aharley/rvl_cdip", split="test")
        
#         # Initialize transforms
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])
        
#         # Load the model
#         self.model = self._load_model(checkpoint_path)
#         logger.info(f"Model loaded successfully on {self.device}")

#     def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
#         """Load the trained model from checkpoint"""
#         from app import MortgageDocumentClassifier  # Import your model class
        
#         model = MortgageDocumentClassifier()
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model = model.to(self.device)
#         model.eval()
#         return model

#     def get_random_samples(self, num_samples: int = 5) -> List[Dict]:
#         """Get random samples from test dataset"""
#         indices = random.sample(range(len(self.dataset)), num_samples)
#         samples = []
        
#         for idx in indices:
#             item = self.dataset[idx]
#             # Convert label tensor to integer and then to string
#             label_idx = item['label']
#             if torch.is_tensor(label_idx):
#                 label_idx = label_idx.item()
#             true_label = self.dataset.features['label'].int2str(label_idx)
            
#             samples.append({
#                 'image': item['image'],
#                 'true_label': true_label,
#                 'index': idx
#             })
        
#         return samples

#     def process_image(self, image) -> torch.Tensor:
#         """Process a single image for model input"""
#         if isinstance(image, dict) and 'bytes' in image:
#             # Handle PIL image from dataset
#             image = Image.open(image['path'] if 'path' in image else image['bytes'])
#         elif isinstance(image, (torch.Tensor, Image.Image)):
#             image = image

#         if isinstance(image, torch.Tensor):
#             image = image.numpy() if image.device.type == 'cpu' else image.cpu().numpy()
#             image = Image.fromarray(image)

#         if image.mode != 'RGB':
#             image = image.convert('RGB')
            
#         return self.transform(image).unsqueeze(0).to(self.device)

#     def predict(self, image) -> Dict:
#         """Make prediction for a single image"""
#         tensor = self.process_image(image)
        
#         with torch.no_grad():
#             output = self.model(tensor)
#             probabilities = output.squeeze().cpu().numpy()
            
#             # Get top predictions
#             top_indices = probabilities.argsort()[-3:][::-1]
#             top_probs = probabilities[top_indices]
            
#             predictions = {
#                 'top_predictions': [
#                     {
#                         'class': self.classes[idx],
#                         'confidence': float(prob),
#                         'above_threshold': prob > self.threshold
#                     }
#                     for idx, prob in zip(top_indices, top_probs)
#                 ],
#                 'raw_probabilities': probabilities.tolist()
#             }
            
#         return predictions

#     def visualize_predictions(self, samples: List[Dict], predictions: List[Dict]):
#         """Visualize the test images with their predictions"""
#         num_samples = len(samples)
#         fig = plt.figure(figsize=(15, 4 * num_samples))
        
#         for idx, (sample, pred) in enumerate(zip(samples, predictions)):
#             # Create subplot
#             ax = plt.subplot(num_samples, 1, idx + 1)
            
#             # Display image
#             image = sample['image']
#             if isinstance(image, dict) and 'bytes' in image:
#                 image = Image.open(image['path'] if 'path' in image else image['bytes'])
#             elif isinstance(image, torch.Tensor):
#                 image = image.numpy() if image.device.type == 'cpu' else image.cpu().numpy()
#                 image = Image.fromarray(image)
            
#             plt.imshow(image)
            
#             # Remove axes
#             plt.axis('off')
            
#             # Add predictions as text
#             title = f"True Label: {sample['true_label']}\n"
#             for p in pred['top_predictions']:
#                 if p['confidence'] > 0.01:  # Only show predictions > 1%
#                     title += f"{p['class']}: {p['confidence']:.1%} "
#                     title += f"({'Above' if p['above_threshold'] else 'Below'} threshold)\n"
            
#             plt.title(title, pad=10)
        
#         plt.tight_layout()
#         plt.savefig('test_predictions.png', bbox_inches='tight', dpi=300)
#         plt.close()
        
#         logger.info("Visualization saved as 'test_predictions.png'")

#     def test_random_samples(self, num_samples: int = 5):
#         """Test model on random samples and display results"""
#         samples = self.get_random_samples(num_samples)
#         all_predictions = []
        
#         for i, sample in enumerate(samples, 1):
#             logger.info(f"\nTesting sample {i}/{num_samples}")
#             logger.info(f"True label: {sample['true_label']}")
            
#             try:
#                 predictions = self.predict(sample['image'])
#                 all_predictions.append(predictions)
                
#                 logger.info("Top predictions:")
#                 for pred in predictions['top_predictions']:
#                     logger.info(
#                         f"- {pred['class']}: {pred['confidence']:.2%} "
#                         f"({'Above' if pred['above_threshold'] else 'Below'} threshold)"
#                     )
#             except Exception as e:
#                 logger.error(f"Error processing sample {i}: {str(e)}")
        
#         # Visualize results
#         self.visualize_predictions(samples, all_predictions)

# def main():
#     # Initialize testing pipeline
#     pipeline = TestingPipeline(
#         checkpoint_path='checkpoints/checkpoint_epoch_9_20250203_150105_acc_0.9381.pt',
#         threshold=0.2
#     )
    
#     # Test random samples with visualization
#     pipeline.test_random_samples(num_samples=5)

# if __name__ == "__main__":
#     main()

import os
import torch
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from datasets import load_dataset
from torchvision import transforms
from typing import List, Dict, Union, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentClassifier:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'mps',
        threshold: float = 0.2
    ):
        self.device = device if torch.backends.mps.is_available() else 'cpu'
        self.threshold = threshold
        self.classes = ['form', 'letter', 'invoice', 'budget', 'specification']
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load the model
        self.model = self._load_model(checkpoint_path)
        logger.info(f"Model loaded successfully on {self.device}")

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load the trained model from checkpoint"""
        from app import MortgageDocumentClassifier
        
        model = MortgageDocumentClassifier()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def process_image(self, image_path: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Process an image for model input"""
        try:
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path)
            elif isinstance(image_path, Image.Image):
                image = image_path
            else:
                raise ValueError("Unsupported image type")

            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def predict(self, image_path: Union[str, Path, Image.Image]) -> Dict:
        """Make prediction for a single image"""
        tensor = self.process_image(image_path)
        
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = output.squeeze().cpu().numpy()
            
            # Get top predictions
            top_indices = probabilities.argsort()[-3:][::-1]
            top_probs = probabilities[top_indices]
            
            predictions = {
                'top_predictions': [
                    {
                        'class': self.classes[idx],
                        'confidence': float(prob),
                        'above_threshold': prob > self.threshold
                    }
                    for idx, prob in zip(top_indices, top_probs)
                ],
                'raw_probabilities': probabilities.tolist()
            }
            
        return predictions

    def classify_documents(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[str] = None,
        batch_size: int = 5
    ) -> List[Dict]:
        """
        Classify documents from a file or directory
        
        Args:
            input_path: Path to a single document or directory of documents
            output_dir: Directory to save visualizations (optional)
            batch_size: Number of images to process in each visualization batch
        """
        input_path = Path(input_path)
        results = []
        
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of files to process
        if input_path.is_file():
            files = [input_path]
        else:
            files = list(input_path.glob('*.png')) + \
                   list(input_path.glob('*.jpg')) + \
                   list(input_path.glob('*.jpeg')) + \
                   list(input_path.glob('*.pdf'))
        
        if not files:
            logger.warning(f"No supported files found in {input_path}")
            return results
        
        # Process files in batches
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_results = []
            
            # Process each file in batch
            for file_path in batch_files:
                try:
                    logger.info(f"Processing {file_path.name}")
                    predictions = self.predict(file_path)
                    result = {
                        'file_path': str(file_path),
                        'predictions': predictions
                    }
                    batch_results.append(result)
                    results.append(result)
                    
                    # Log predictions
                    logger.info("Top predictions:")
                    for pred in predictions['top_predictions']:
                        logger.info(
                            f"- {pred['class']}: {pred['confidence']:.2%} "
                            f"({'Above' if pred['above_threshold'] else 'Below'} threshold)"
                        )
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
            
            # Create visualization for batch
            if output_dir and batch_results:
                self.visualize_batch(batch_results, output_dir)
        
        return results

    def visualize_batch(self, results: List[Dict], output_dir: Path):
        """Create visualization for a batch of results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_images = len(results)
        fig = plt.figure(figsize=(15, 4 * num_images))
        
        for idx, result in enumerate(results):
            # Create subplot
            ax = plt.subplot(num_images, 1, idx + 1)
            
            # Display image
            image = Image.open(result['file_path'])
            plt.imshow(image)
            plt.axis('off')
            
            # Add predictions as text
            title = f"File: {Path(result['file_path']).name}\n"
            for pred in result['predictions']['top_predictions']:
                if pred['confidence'] > 0.01:  # Only show predictions > 1%
                    title += f"{pred['class']}: {pred['confidence']:.1%} "
                    title += f"({'Above' if pred['above_threshold'] else 'Below'} threshold)\n"
            
            plt.title(title, pad=10)
        
        plt.tight_layout()
        output_path = output_dir / f'predictions_{timestamp}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Visualization saved as '{output_path}'")

def main():
    # Initialize classifier
    classifier = DocumentClassifier(
        checkpoint_path='checkpoints/checkpoint_epoch_9_20250203_150105_acc_0.9381.pt',
        threshold=0.2
    )
    
    # Example usage for custom documents
    custom_path = 'Phanee.png'  # Change this to your documents path
    output_dir = 'predictions'
    
    results = classifier.classify_documents(
        input_path=custom_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()