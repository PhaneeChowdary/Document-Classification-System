import os
import torch
import mlflow
import logging
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Tuple
from datasets import load_dataset
from pytesseract import image_to_string
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define mortgage-relevant document types
MORTGAGE_RELEVANT = {
    'form': [1, 0, 0, 0, 0],        # Mortgage forms
    'letter': [0, 1, 0, 0, 0],      # Approval letters
    'invoice': [0, 0, 1, 0, 0],     # Payment records
    'budget': [0, 0, 0, 1, 0],      # Financial statements
    'specification': [0, 0, 0, 0, 1] # Property specs
}

class MortgageDocumentDataset(Dataset):
    """
    Dataset for mortgage document classification
    """
    def __init__(self, split: str = "train"):
        self.dataset = load_dataset("aharley/rvl_cdip", split=split).with_format("torch")
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _preprocess_image(self, image):
        """Helper method to preprocess the image"""
        if torch.is_tensor(image):
            image_np = image.numpy()
            if len(image_np.shape) == 3 and image_np.shape[0] == 1:
                image_np = image_np.squeeze(0)
            image = Image.fromarray(image_np)

        # Convert to RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        # Preprocess image
        image = self._preprocess_image(image)
        image = self.transform(image)

        # Handle the label
        if torch.is_tensor(label):
            label = label.tolist()
        label_str = self.dataset.features['label'].int2str(label)
        multi_label = MORTGAGE_RELEVANT.get(label_str, [0, 0, 0, 0, 0])

        return image, torch.tensor(multi_label, dtype=torch.float32)

class MortgageDocumentClassifier(nn.Module):
    """
    Document classifier using Vision Transformer
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return torch.sigmoid(self.vit(x))

def extract_text(image: Image.Image) -> str:
    """
    Extract text from an image using Tesseract OCR
    """
    return image_to_string(image)

def create_data_loaders(batch_size: int = 32) -> Dict[str, DataLoader]:
    """
    Create data loaders for training and validation
    """
    train_dataset = MortgageDocumentDataset(split="train")
    val_dataset = MortgageDocumentDataset(split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=True,  # Add this line
        prefetch_factor=2  # Add this line
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=True,  # Add this line
        prefetch_factor=2  # Add this line
    )

    return {
        'train': train_loader,
        'val': val_loader
    }

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch+1} Training', leave=False, dynamic_ncols=True)

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).all(dim=1).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'accuracy': f'{correct/total:.4f}'
        })

    return total_loss / len(loader), correct / total

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False, dynamic_ncols=True)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).all(dim=1).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{correct/total:.4f}'
            })

    return total_loss / len(loader), correct / total

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    checkpoint_dir: str = "checkpoints"
):
    """
    Save model checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(
        checkpoint_dir, 
        f'checkpoint_epoch_{epoch}_{timestamp}_acc_{accuracy:.4f}.pt'
    )
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }, filename)
    
    logger.info(f"Saved checkpoint: {filename}")

def main():
    # Clean up any zombie processes
    torch.multiprocessing.set_start_method('spawn', force=True)

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mortgage-document-classification")
    
    with mlflow.start_run():
        # Setup
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Model initialization
        model = MortgageDocumentClassifier(num_classes=5).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # MPS-specific optimizations
        if device.type == 'mps':
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.5)

        # Create data loaders
        loaders = create_data_loaders(batch_size=16)
        
        # Log parameters
        mlflow.log_params({
            "model_type": "ViT-B-16",
            "optimizer": "AdamW",
            "learning_rate": 1e-5,
            "batch_size": 16
        })
        
        # Training loop
        num_epochs = 10
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = train_one_epoch(
                model, loaders['train'], criterion, optimizer, device, epoch
            )
            
            # Validate
            val_loss, val_acc = validate(
                model, loaders['val'], criterion, device
            )
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)
            
            logger.info(
                f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                save_checkpoint(model, optimizer, epoch, val_acc)

if __name__ == "__main__":
    main()