import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path

# Import our modules - use absolute imports
from .models import load_model, save_model
from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric

def train_classification():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50
    num_workers = 2
    
    # Data paths - updated for Colab structure
    train_path = "classification_data/train"
    val_path = "classification_data/val"
    
    # Load data
    print("Loading training data...")
    train_loader = load_data(
        train_path, 
        transform_pipeline="aug",  # Use augmentation for training
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    print("Loading validation data...")
    val_loader = load_data(
        val_path, 
        transform_pipeline="default",  # No augmentation for validation
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Create model
    print("Creating model...")
    model = load_model("classifier", in_channels=3, num_classes=6)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Metrics
    accuracy_metric = AccuracyMetric()
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%')
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
            save_model(model)
            print(f'Model saved to: {Path("classifier.th")}')
        
        # Early stopping if accuracy is good enough
        if val_acc > 85.0:
            print(f'Reached target accuracy of 85%! Stopping early.')
            break
    
    print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    # Final evaluation
    model.eval()
    final_val_acc = 0.0
    val_total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            val_total += target.size(0)
            final_val_acc += predicted.eq(target).sum().item()
    
    final_val_acc = 100. * final_val_acc / val_total
    print(f'Final validation accuracy: {final_val_acc:.2f}%')

if __name__ == "__main__":
    train_classification()
