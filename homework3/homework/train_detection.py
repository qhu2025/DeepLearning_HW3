import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from pathlib import Path

# Import our modules - use absolute imports
from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import ConfusionMatrix

class FocalLoss(nn.Module):
    """Focal Loss to handle extreme class imbalance"""
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train_detection():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 🔧 OPTIMIZED FOR ENHANCED ARCHITECTURE
    batch_size = 12  # Slightly smaller for deeper model
    learning_rate = 0.0001  # Higher LR for deeper model with attention
    num_epochs = 30  # Fewer epochs needed with better architecture
    num_workers = 2
    
    # Loss weights - optimized for enhanced model
    seg_weight = 3.0  # Higher segmentation focus
    depth_weight = 1.0  # Balanced depth learning
    
    # Data paths
    train_path = "drive_data/train"
    val_path = "drive_data/val"
    
    # Load data
    print("Loading training data...")
    train_loader = load_data(
        train_path, 
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    print("Loading validation data...")
    val_loader = load_data(
        val_path, 
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Create model
    print("Creating model...")
    model = load_model("detector", in_channels=3, num_classes=3)
    model = model.to(device)
    
    # 🔧 BALANCED WEIGHTS for Enhanced Architecture
    class_weights = torch.tensor([1.0, 5.0, 5.0]).to(device)  # Balanced for attention model
    print(f"✅ ENHANCED ARCHITECTURE WEIGHTS: {class_weights}")
    print("   (Background: 1.0, Left Lane: 5.0, Right Lane: 5.0)")
    print("   🎯 Target: IoU 0.75+ with attention mechanisms!")
    
    # Standard losses
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights)
    depth_criterion = nn.L1Loss()
    
    # Enhanced optimizer for deeper model
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate/10)
    
    # Metrics
    confusion_matrix = ConfusionMatrix(num_classes=3)
    
    # Training loop
    best_mean_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_seg_loss = 0.0
        train_depth_loss = 0.0
        train_total_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            tracks = batch['track'].to(device)
            
            optimizer.zero_grad()
            
            seg_logits, pred_depths = model(images)
            
            seg_loss = seg_criterion(seg_logits, tracks)
            depth_loss = depth_criterion(pred_depths, depths)
            total_loss = seg_weight * seg_loss + depth_weight * depth_loss
            
            total_loss.backward()
            
            # Gentle gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            train_seg_loss += seg_loss.item()
            train_depth_loss += depth_loss.item()
            train_total_loss += total_loss.item()
            
            if batch_idx % 100 == 0:  # Less frequent logging
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Seg Loss: {seg_loss.item():.4f}, Depth Loss: {depth_loss.item():.4f}, '
                      f'Total Loss: {total_loss.item():.4f}')
        
        avg_train_seg_loss = train_seg_loss / len(train_loader)
        avg_train_depth_loss = train_depth_loss / len(train_loader)
        avg_train_total_loss = train_total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_seg_loss = 0.0
        val_depth_loss = 0.0
        val_total_loss = 0.0
        
        confusion_matrix.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                depths = batch['depth'].to(device)
                tracks = batch['track'].to(device)
                
                seg_logits, pred_depths = model(images)
                
                seg_loss = seg_criterion(seg_logits, tracks)
                depth_loss = depth_criterion(pred_depths, depths)
                total_loss = seg_weight * seg_loss + depth_weight * depth_loss
                
                val_seg_loss += seg_loss.item()
                val_depth_loss += depth_loss.item()
                val_total_loss += total_loss.item()
                
                pred_seg = seg_logits.argmax(dim=1)
                confusion_matrix.add(pred_seg.cpu(), tracks.cpu())
        
        avg_val_seg_loss = val_seg_loss / len(val_loader)
        avg_val_depth_loss = val_depth_loss / len(val_loader)
        avg_val_total_loss = val_total_loss / len(val_loader)
        
        metrics = confusion_matrix.compute()
        mean_iou = metrics.get('mean_iou', 0.0)
        iou_per_class = metrics.get('iou', [0.0, 0.0, 0.0])
        
        # Handle case where iou might be a single float instead of list
        if isinstance(iou_per_class, (int, float)):
            iou_per_class = [iou_per_class, 0.0, 0.0]
        elif not isinstance(iou_per_class, (list, tuple)):
            iou_per_class = [0.0, 0.0, 0.0]
        
        # Use cosine annealing scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s')
        print(f'Train - Seg Loss: {avg_train_seg_loss:.4f}, Depth Loss: {avg_train_depth_loss:.4f}, Total: {avg_train_total_loss:.4f}')
        print(f'Val - Seg Loss: {avg_val_seg_loss:.4f}, Depth Loss: {avg_val_depth_loss:.4f}, Total: {avg_val_total_loss:.4f}')
        print(f'Validation mIoU: {mean_iou:.4f}')
        if len(iou_per_class) >= 3:
            print(f'Per-class IoU - Background: {iou_per_class[0]:.3f}, Left: {iou_per_class[1]:.3f}, Right: {iou_per_class[2]:.3f}')
        
        print('-' * 80)
        
        # 🔧 ALWAYS save if this is the best model so far
        if mean_iou >= best_mean_iou:
            best_mean_iou = mean_iou
            print(f'💾 Saving model with mIoU: {best_mean_iou:.4f}')
            save_model(model)
            print(f'Model saved to: {Path("detector.th")}')
        
        # Early stopping if target reached
        if mean_iou > 0.75:
            print(f'🎯 TARGET REACHED! mIoU: {mean_iou:.4f} > 0.75')
            break
    
    # 🔧 CRITICAL: ALWAYS save model at the end, regardless of performance
    print(f'\n💾 FORCE SAVING final model...')
    save_model(model)
    print(f'✅ detector.th saved successfully!')
    
    print(f'Training completed! Best mIoU: {best_mean_iou:.4f}')
    
    # Final evaluation
    model.eval()
    confusion_matrix.reset()
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            tracks = batch['track'].to(device)
            
            seg_logits, _ = model(images)
            pred_seg = seg_logits.argmax(dim=1)
            confusion_matrix.add(pred_seg.cpu(), tracks.cpu())
    
    final_metrics = confusion_matrix.compute()
    final_mean_iou = final_metrics.get('mean_iou', 0.0)
    final_iou_per_class = final_metrics.get('iou', [0.0, 0.0, 0.0])
    
    # Handle case where iou might be a single float instead of list
    if isinstance(final_iou_per_class, (int, float)):
        final_iou_per_class = [final_iou_per_class, 0.0, 0.0]
    elif not isinstance(final_iou_per_class, (list, tuple)):
        final_iou_per_class = [0.0, 0.0, 0.0]
    
    print(f'\n🏁 FINAL RESULTS:')
    print(f'Final mIoU: {final_mean_iou:.4f}')
    if len(final_iou_per_class) >= 3:
        print(f'Final per-class IoU:')
        print(f'  Background: {final_iou_per_class[0]:.3f}')
        print(f'  Left Lane:  {final_iou_per_class[1]:.3f}')
        print(f'  Right Lane: {final_iou_per_class[2]:.3f}')
    
    if final_mean_iou >= 0.75:
        print('✅ SUCCESS: Meets grading requirements!')
    else:
        print('❌ BELOW TARGET: But model is saved for grader!')
    
    # 🔧 Double-check the file exists
    if os.path.exists("detector.th"):
        print(f'✅ Confirmed: detector.th exists and ready for grader!')
    else:
        print(f'❌ ERROR: detector.th not found! Trying to save again...')
        save_model(model)

if __name__ == "__main__":
    train_detection() 
