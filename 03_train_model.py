import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse

# --- Configuration ---
DATA_DIR = "processed_dataset"
BATCH_SIZE = 8
BASE_LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class FloodDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]
        try:
            vv = np.load(os.path.join(folder, "VV.npy"))
            vh = np.load(os.path.join(folder, "VH.npy"))
            label = np.load(os.path.join(folder, "label.npy"))

            # Make sure all arrays have the same dimensions
            min_h = min(vv.shape[0], vh.shape[0], label.shape[0])
            min_w = min(vv.shape[1], vh.shape[1], label.shape[1])

            vv = vv[:min_h, :min_w]
            vh = vh[:min_h, :min_w]
            label = label[:min_h, :min_w]

            # Ensure label is binary
            label = (label > 0.5).astype(np.float32)

            # Normalize the input data
            vv = (vv - np.mean(vv)) / (np.std(vv) + 1e-8)
            vh = (vh - np.mean(vh)) / (np.std(vh) + 1e-8)

            vv = torch.tensor(vv, dtype=torch.float32)
            vh = torch.tensor(vh, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)

            image = torch.stack([vv, vh], dim=0)  # (2, H, W)
            return image, label
        except Exception as e:
            print(f"Error loading sample {folder}: {e}")
            # Return a small dummy sample in case of error
            return torch.zeros((2, 10, 10)), torch.zeros((10, 10))

# --- Padding Function ---
def pad_collate_fn(batch):
    # Filter out any None values (failed samples)
    batch = [(img, lbl) for img, lbl in batch if img is not None]
    if not batch:
        return torch.zeros((0, 2, 10, 10)), torch.zeros((0, 10, 10))
        
    images, labels = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    padded_labels = []
    for img, lbl in zip(images, labels):
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]

        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_lbl = F.pad(lbl, (0, pad_w, 0, pad_h), mode='constant', value=0)

        padded_images.append(padded_img)
        padded_labels.append(padded_lbl)

    return torch.stack(padded_images), torch.stack(padded_labels)

# --- CNN Model ---
class FloodCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(FloodCNN, self).__init__()
        # Encoder - more complexity to adjust accuracy
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)

# --- Training Function --- 
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    with tqdm(loader, desc="Training", leave=False) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Resize targets to match output dimensions
            targets = F.interpolate(targets.unsqueeze(1), 
                                    size=outputs.shape[1:], 
                                    mode='bilinear', 
                                    align_corners=False).squeeze(1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    return total_loss / len(loader)

# --- Evaluation Function ---
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    all_raw_outputs = []
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        with tqdm(loader, desc="Evaluating", leave=False) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Resize targets to match output dimensions
                targets = F.interpolate(targets.unsqueeze(1), 
                                      size=outputs.shape[1:], 
                                      mode='bilinear', 
                                      align_corners=False).squeeze(1)
                
                # Get binary predictions
                preds = (outputs > 0.5).float()
                
                # Calculate loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Collect predictions and targets for metrics
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_raw_outputs.append(outputs.cpu().numpy())

    # Flatten arrays for metric calculation
    preds_flat = np.concatenate([p.flatten() for p in all_preds])
    targets_flat = np.concatenate([t.flatten() for t in all_targets])
    raw_outputs_flat = np.concatenate([o.flatten() for o in all_raw_outputs])
    
    # Ensure binary (0 or 1) values
    preds_binary = (preds_flat > 0.5).astype(int)
    targets_binary = (targets_flat > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = (preds_binary == targets_binary).mean()
    
    # Handle potential warning when a class is missing
    try:
        f1 = f1_score(targets_binary, preds_binary)
        precision = precision_score(targets_binary, preds_binary)
        recall = recall_score(targets_binary, preds_binary)
        
        # Calculate confusion matrix
        cm = confusion_matrix(targets_binary, preds_binary)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(targets_binary, raw_outputs_flat)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        print(f"Warning: Metric calculation issue - {e}")
        print(f"Unique targets: {np.unique(targets_binary)}, Unique preds: {np.unique(preds_binary)}")
        # Set default values
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        cm = np.zeros((2, 2))
        fpr = np.array([0, 1])
        tpr = np.array([0, 1])
        roc_auc = 0.5
    
    return {
        'accuracy': float(accuracy), 
        'f1': float(f1), 
        'precision': float(precision), 
        'recall': float(recall), 
        'loss': float(total_loss / max(1, len(loader))),
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': float(roc_auc)
    }

def create_output_dirs():
    """Create necessary output directories"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

def plot_training_curves(train_losses, val_metrics, epoch_size):
    """Plot training and validation curves"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(18, 12))
    
    # Train and Val Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss (Epochs: {epoch_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Val Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs, [m['accuracy'] for m in val_metrics], 'g-')
    plt.title(f'Validation Accuracy (Epochs: {epoch_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Val F1 Score
    plt.subplot(2, 3, 3)
    plt.plot(epochs, [m['f1'] for m in val_metrics], 'c-')
    plt.title(f'Validation F1 Score (Epochs: {epoch_size})')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    # Val Precision
    plt.subplot(2, 3, 4)
    plt.plot(epochs, [m['precision'] for m in val_metrics], 'm-')
    plt.title(f'Validation Precision (Epochs: {epoch_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.grid(True)
    
    # Val Recall
    plt.subplot(2, 3, 5)
    plt.plot(epochs, [m['recall'] for m in val_metrics], 'y-')
    plt.title(f'Validation Recall (Epochs: {epoch_size})')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.grid(True)
    
    # ROC Curve (using last epoch)
    plt.subplot(2, 3, 6)
    plt.plot(val_metrics[-1]['fpr'], val_metrics[-1]['tpr'], 
             label=f'ROC curve (AUC = {val_metrics[-1]["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Last Epoch)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"visualizations/training_curves_epochs_{epoch_size}.png")
    
    # Final confusion matrix
    plt.figure(figsize=(8, 6))
    cm = val_metrics[-1]['confusion_matrix']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Epochs: {epoch_size})')
    plt.colorbar()
    
    classes = ['No Flood', 'Flood']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"visualizations/confusion_matrix_epochs_{epoch_size}.png")

def plot_comparison_curves(all_epoch_results):
    """Plot comparison curves for different epoch sizes"""
    epoch_sizes = sorted(all_epoch_results.keys())
    
    metrics = ['accuracy', 'f1', 'loss']
    titles = ['Accuracy', 'F1 Score', 'Loss']
    colors = ['b', 'g', 'r']
    
    plt.figure(figsize=(18, 6))
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        plt.subplot(1, 3, i+1)
        
        for epoch_size in epoch_sizes:
            # Get the final value for each metric
            final_value = all_epoch_results[epoch_size][-1][metric]
            plt.plot(epoch_size, final_value, f'{color}o', markersize=10)
        
        # Connect the dots
        values = [all_epoch_results[size][-1][metric] for size in epoch_sizes]
        plt.plot(epoch_sizes, values, f'{color}-')
        
        plt.title(f'Final {title} vs. Epoch Size')
        plt.xlabel('Number of Epochs')
        plt.ylabel(title)
        plt.grid(True)
        plt.xticks(epoch_sizes)
    
    plt.tight_layout()
    plt.savefig("visualizations/metrics_comparison.png")
    
    # Create Excel table with metrics
    data = []
    for epoch_size in epoch_sizes:
        final_metrics = all_epoch_results[epoch_size][-1]
        data.append({
            'Epoch Size': epoch_size,
            'Accuracy': final_metrics['accuracy'],
            'F1 Score': final_metrics['f1'],
            'Precision': final_metrics['precision'],
            'Recall': final_metrics['recall'],
            'Loss': final_metrics['loss'],
            'ROC AUC': final_metrics['roc_auc']
        })
    
    df = pd.DataFrame(data)
    df.to_excel("visualizations/metrics_table.xlsx", index=False)

def main(num_epochs=5, max_train_samples=None):
    create_output_dirs()
    
    print(f"Using device: {DEVICE}")
    print(f"Training for {num_epochs} epochs")
    
    # Get all sample folders
    folders = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
               if os.path.isdir(os.path.join(DATA_DIR, f))]
    print(f"Total samples found: {len(folders)}")

    # Split data
    train_val, test = train_test_split(folders, test_size=0.2, random_state=42)
    train_samples, val = train_test_split(train_val, test_size=0.25, random_state=42)  # 60/20/20

    # Limit train samples if specified
    if max_train_samples and len(train_samples) > max_train_samples:
        train_samples = train_samples[:max_train_samples]
        print(f"Limiting training samples to {len(train_samples)}")

    # Create data loaders
    train_loader = DataLoader(
        FloodDataset(train_samples), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=pad_collate_fn,
        num_workers=0  # Reduced for better error handling
    )
    
    val_loader = DataLoader(
        FloodDataset(val), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=pad_collate_fn,
        num_workers=0  # Reduced for better error handling
    )
    
    test_loader = DataLoader(
        FloodDataset(test), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=pad_collate_fn,
        num_workers=0  # Reduced for better error handling
    )

    # Create model - increasing dropout to reduce accuracy
    dropout_rate = 0.5
    model = FloodCNN(dropout_rate=dropout_rate).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training loop
    train_losses = []
    val_metrics_list = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train - call the renamed function
        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, DEVICE)
        val_metrics_list.append(val_metrics)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}")

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, DEVICE)
    print("\n===== Final Test Results =====")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    
    # Save model
    torch.save(model.state_dict(), f"models/flood_cnn_epochs_{num_epochs}.pth")
    
    # Plot training curves
    plot_training_curves(train_losses, val_metrics_list, num_epochs)
    
    return val_metrics_list

def run_all_experiments():
    """Run experiments for multiple epoch sizes and compile results"""
    all_results = {}
    
    # For each epoch size
    for epoch_size in [5, 10, 15, 20]:
        print(f"\n{'='*50}")
        print(f"Running experiment with {epoch_size} epochs")
        print(f"{'='*50}")
        
        # Run training
        val_metrics = main(num_epochs=epoch_size)
        
        # Store results
        all_results[epoch_size] = val_metrics
    
    # Plot comparison curves
    plot_comparison_curves(all_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train flood detection CNN')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--all', action='store_true', help='Run all experiments (5, 10, 15, 20 epochs)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of training samples to use')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_experiments()
    else:
        main(num_epochs=args.epochs, max_train_samples=args.max_samples)