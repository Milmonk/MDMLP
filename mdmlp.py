import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import einops
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Overlap Patch Embedding
class OverlapPatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, overlap_size=2, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        stride = patch_size - overlap_size
        self.num_patches_h = (img_size - patch_size) // stride + 1
        self.num_patches_w = (img_size - patch_size) // stride + 1
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=0)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv(x)  # [B, embed_dim, H', W']
        x = einops.rearrange(x, 'b c h w -> b h w c')  # [B, H', W', embed_dim]
        x = self.norm(x)
        return x  # [B, H', W', embed_dim]

# MDLayer
class MDLayer(nn.Module):
    def __init__(self, dim, op_dim, expansion_factor=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(op_dim, op_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(op_dim * expansion_factor, op_dim),
            nn.Dropout(0.1)
        )
        self.op_dim = op_dim

    def forward(self, x, transpose_dims):
        # x: [B, H', W', D]
        x_norm = self.norm(x)
        # Transpose to make op_dim the last dimension
        x_t = einops.rearrange(x_norm, transpose_dims[0])
        x_t = self.mlp(x_t)
        # Transpose back
        x_t = einops.rearrange(x_t, transpose_dims[1])
        return x + x_t  # Residual connection

# MDBlock
class MDBlock(nn.Module):
    def __init__(self, h, w, d, expansion_factor=4):
        super().__init__()
        self.height_layer = MDLayer(d, h, expansion_factor)
        self.width_layer = MDLayer(d, w, expansion_factor)
        self.token_layer = MDLayer(d, d, expansion_factor)

    def forward(self, x):
        # x: [B, H', W', D]
        x = self.height_layer(x, (
            'b h w d -> b w d h',
            'b w d h -> b h w d'
        ))
        x = self.width_layer(x, (
            'b h w d -> b h d w',
            'b h d w -> b h w d'
        ))
        x = self.token_layer(x, (
            'b h w d -> b h w d',
            'b h w d -> b h w d'
        ))
        return x

# MDAttnTool
class MDAttnTool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mlp_w = nn.Sequential(
            nn.Linear(channels, 8),
            nn.GELU(),
            nn.Linear(8, 1)
        )
        self.mlp_h = nn.Sequential(
            nn.Linear(1, 8),  # Input is 1 channel from mlp_w
            nn.GELU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Normalize
        x_norm = einops.rearrange(x, 'b c h w -> b h w c')  # [B, H, W, C]
        x_norm = self.norm(x_norm)  # [B, H, W, C]
        # Width features
        y1 = self.mlp_w(x_norm)  # [B, H, W, 1]
        y1 = einops.rearrange(y1, 'b h w 1 -> b 1 h w')  # [B, 1, H, W]
        # Height features
        y2 = einops.rearrange(y1, 'b 1 h w -> b w h 1')  # [B, W, H, 1]
        y2 = self.mlp_h(y2)  # [B, W, H, 1]
        v = einops.rearrange(y2, 'b w h 1 -> b 1 h w')  # [B, 1, H, W]
        # Element-wise multiplication
        y = x * v  # [B, C, H, W] * [B, 1, H, W] -> [B, C, H, W]
        return y, v

# MDMLP
class MDMLP(nn.Module):
    def __init__(self, img_size=32, patch_size=4, overlap_size=2, in_channels=3, num_classes=10,
                 embed_dim=64, depth=8, expansion_factor=4):
        super().__init__()
        self.patch_embed = OverlapPatchEmbedding(img_size, patch_size, overlap_size, in_channels, embed_dim)
        self.num_patches_h = self.patch_embed.num_patches_h
        self.num_patches_w = self.patch_embed.num_patches_w
        self.blocks = nn.ModuleList([
            MDBlock(self.num_patches_h, self.num_patches_w, embed_dim, expansion_factor)
            for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )
        self.attn_tool = MDAttnTool(in_channels)

    def forward(self, x):
        # x: [B, C, H, W]
        # Attention visualization
        attn_output, weights = self.attn_tool(x)
        # Patch embedding
        x = self.patch_embed(x)  # [B, H', W', D]
        # MDBlocks
        for block in self.blocks:
            x = block(x)
        # Classification head
        x = einops.rearrange(x, 'b h w d -> b d h w')
        x = self.head(x)
        return x, weights

# Visualization function for MDAttnTool weights
def visualize_attention(images, attn_weights, epoch, save_dir="attention_maps"):
    os.makedirs(save_dir, exist_ok=True)
    images = images.cpu().numpy()  # [B, C, H, W]
    attn_weights = attn_weights.cpu().numpy()  # [B, 1, H, W]
    
    # Denormalize images (CIFAR-10 normalization: mean=0.5, std=0.5)
    images = images * 0.5 + 0.5  # [B, C, H, W]
    images = np.clip(images, 0, 1)
    
    for i in range(min(5, images.shape[0])):  # Visualize up to 5 images
        img = images[i].transpose(1, 2, 0)  # [H, W, C]
        attn = attn_weights[i, 0]  # [H, W]
        
        # Normalize attention weights for visualization
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        plt.figure(figsize=(8, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")
        
        # Attention heatmap overlaid on image
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(attn, cmap="jet", alpha=0.5)
        plt.title("Attention Map")
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_image_{i}.png"))
        plt.close()
    
    print(f"Saved attention maps for epoch {epoch} in {save_dir}/", flush=True)

# Plot training loss and accuracy
def plot_metrics(losses, accuracies, save_path="training_metrics.png"):
    plt.figure(figsize=(10, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracies) + 1), accuracies, label="Test Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy Over Epochs")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training metrics plot to {save_path}", flush=True)

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training function
def train_model(model, train_loader, test_loader, device, epochs=200):
    print("Starting training...", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_batches = len(train_loader)
    print(f"Total batches per epoch: {total_batches}", flush=True)
    
    # Track metrics
    epoch_losses = []
    epoch_accuracies = []
    
    print("\nEpoch | Avg Loss | Test Accuracy")
    print("-" * 35)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} started...", flush=True)
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 1):
            # print(f"Processing batch {i}/{total_batches}...", flush=True)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 0:  # Print every 50 batches
                print(f"Batch {i}/{total_batches}, Loss: {loss.item():.4f}", flush=True)
        
        avg_loss = running_loss / total_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}", flush=True)
        scheduler.step()

        # Evaluate
        print("Evaluating model...", flush=True)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader, 1):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, attn_weights = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if i % 50 == 0:
                    print(f"Test batch {i}/{len(test_loader)} processed", flush=True)
                # Visualize attention for the first batch of the last epoch
                if epoch == epochs - 1 and i == 1:
                    visualize_attention(inputs, attn_weights, epoch + 1)
        accuracy = 100 * correct / total
        epoch_accuracies.append(accuracy)
        print(f"Epoch {epoch+1} Accuracy: {accuracy:.2f}%", flush=True)
        
        # Print table row
        print(f"{epoch+1:5d} | {avg_loss:8.4f} | {accuracy:12.2f}%")
    
    # Final evaluation
    print("\nFinal Evaluation...", flush=True)
    model.eval()
    all_preds = []
    all_labels = []
    final_loss = 0.0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            final_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += inputs.size(0)
    
    final_loss = final_loss / total
    final_accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    # Compute precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Model parameters
    num_params = count_parameters(model)
    
    # Hyperparameters
    hyperparams = {
        "Epochs": epochs,
        "Batch Size": 128,
        "Learning Rate": 0.1,
        "Momentum": 0.9,
        "Weight Decay": 0.0001,
        "Optimizer": "SGD",
        "Scheduler": "CosineAnnealingLR",
        "Image Size": 32,
        "Patch Size": 4,
        "Overlap Size": 2,
        "Embed Dim": 64,
        "Depth": 8,
        "Expansion Factor": 4
    }
    
    # Plot metrics
    plot_metrics(epoch_losses, epoch_accuracies)
    
    # Save model
    model_path = "mdmlp_cifar10.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}", flush=True)
    
    # Print final summary table
    print("\nFinal Summary Table")
    print("=" * 50)
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-" * 50)
    print(f"{'Final Test Accuracy':<30} | {final_accuracy:.2f}%")
    print(f"{'Final Test Loss':<30} | {final_loss:.4f}")
    print(f"{'Average Precision':<30} | {avg_precision:.4f}")
    print(f"{'Average Recall':<30} | {avg_recall:.4f}")
    print(f"{'Average F1-Score':<30} | {avg_f1:.4f}")
    print(f"{'Total Parameters':<30} | {num_params:,}")
    print("-" * 50)
    
    # Print per-class metrics
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print("\nPer-Class Metrics")
    print("-" * 70)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for i, cls in enumerate(cifar10_classes):
        print(f"{cls:<12} | {precision[i]:<10.4f} | {recall[i]:<10.4f} | {f1[i]:<10.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix")
    print("-" * 50)
    print("Rows: True Labels, Columns: Predicted Labels")
    print(f"{'':<12}", end="")
    for cls in cifar10_classes:
        print(f"{cls[:3]:<5}", end="")
    print()
    for i, cls in enumerate(cifar10_classes):
        print(f"{cls:<12}", end="")
        for j in range(10):
            print(f"{cm[i,j]:<5}", end="")
        print()
    
    # Print hyperparameters
    print("\nHyperparameters")
    print("-" * 50)
    for key, value in hyperparams.items():
        print(f"{key:<30} | {value}")

# Main
def main():
    print("Initializing script...", flush=True)
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Data transforms (minimal augmentation as per paper)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    print("Loading CIFAR-10 training dataset...", flush=True)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print("Training dataset loaded.", flush=True)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    print(f"Training DataLoader created with {len(train_loader)} batches.", flush=True)

    # Load CIFAR-10 test dataset
    print("Loading CIFAR-10 test dataset...", flush=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    print("Test dataset loaded.", flush=True)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    print(f"Test DataLoader created with {len(test_loader)} batches.", flush=True)

    # Initialize model
    print("Initializing MDMLP model...", flush=True)
    model = MDMLP(
        img_size=32,
        patch_size=4,
        overlap_size=2,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=8,
        expansion_factor=4
    ).to(device)
    print("Model initialized and moved to device.", flush=True)

    # Train
    print("Starting model training...", flush=True)
    train_model(model, train_loader, test_loader, device)

if __name__ == '__main__':
    main()