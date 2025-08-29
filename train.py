import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.dataloader import HAM10000
from data.data_manager import load_metadata, init_db
from model.vit import VisionTransformer
import os
import json
import numpy as np
from torch.utils.data import WeightedRandomSampler

BEST_VAL_ACC_PATH = 'best_acc.json'

def save_best_acc(val_acc):
    with open(BEST_VAL_ACC_PATH, 'w') as f:
        json.dump({"best_acc": val_acc}, f)

def load_best_acc():
    if os.path.exists(BEST_VAL_ACC_PATH):
        with open(BEST_VAL_ACC_PATH, 'r') as f:
            data = json.load(f)
            return data.get("best_acc", 0)
    return 0


def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.0003):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.to(device)

    best_acc = load_best_acc()

    if os.path.exists('vit.pth'):
        model.load_state_dict(torch.load('vit.pth'))
    else:
        print(f"Path 'vit.pth' does not exist yet")

    for epoch in range(epochs):
        model.train()

        train_loss = 0
        
        for (images, labels) in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Epoch {epoch+1}\nAvg Train Loss: {train_loss/len(train_loader):.4f}')
        val_acc = validate(model, val_loader, device)
        if val_acc > best_acc:
            print(f'New best validation accuracy: {val_acc:.4f}% (previous: {best_acc:.4f}%)')
            best_acc = val_acc
            save_best_acc(best_acc)
            torch.save(model.state_dict(), 'vit.pth')
    
def validate(model, val_loader, device):
    model.eval()

    val_loss = 0
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)

    avg_loss = val_loss / len(val_loader)
    val_acc = (correct/total) * 100 if total > 0 else 0

    print(f'Validation Loss: {avg_loss:.4f}\nValidation Accuracy: {val_acc:.4f}%')
    return val_acc
            
    
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    session = init_db()

    dataset_path = load_metadata(session)
    dataset = HAM10000(dataset_path=dataset_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]
    #print("Train class distribution:", np.bincount(train_labels))
    #print("Val class distribution:", np.bincount(val_labels))

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size, num_workers=2)

    model = VisionTransformer(img_shape=224, patch_size=16, depth=6, hidden_dim=256, num_heads=4, mlp_dim=512,num_classes=7)

    try:
        train_model(model, train_loader, val_loader, device)
    except Exception as e:
        print(f"Training error: {e}")





