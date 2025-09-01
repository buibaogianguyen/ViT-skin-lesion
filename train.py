import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.dataloader import ISIC2019
from data.data_manager import load_metadata, init_db
from model.vit import VisionTransformer
import os
import json
import numpy as np
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import timm

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


def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.0003, grad_clip=1.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
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
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

    num_classes = 9
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    avg_loss = val_loss / len(val_loader)
    val_acc = (correct / total) * 100 if total > 0 else 0


    print(f'Validation Loss: {avg_loss:.4f}\nValidation Accuracy: {val_acc:.4f}%')
    for i in range(num_classes):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'  Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})')

    return val_acc

def balance_training_data(train_df, data_aug_rate):
    balanced_dfs = []

    for class_idx, rate in enumerate(data_aug_rate):
        class_df = train_df[train_df['label_idx'] == class_idx]
        if len(class_df) > 0 and rate > 0:
            augmented_df = pd.concat([class_df] * rate, ignore_index=True)
            balanced_dfs.append(augmented_df)

    return pd.concat(balanced_dfs, ignore_index=True)

def balance_val_data(val_df):
    min_count = val_df['label_idx'].value_counts().min()
    balanced_dfs = []
    for class_idx in val_df['label_idx'].unique():
        class_df = val_df[val_df['label_idx'] == class_idx]
        balanced_dfs.append(class_df.sample(min_count, random_state=42))
        
    return pd.concat(balanced_dfs, ignore_index=True)
            
    
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
    dataset = ISIC2019(dataset_path=dataset_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]
    #print("Train class distribution:", np.bincount(train_labels))
    #print("Val class distribution:", np.bincount(val_labels))

    class_counts = np.bincount(train_labels, minlength=9)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)

    model = VisionTransformer(img_shape=224, patch_size=16, depth=6, hidden_dim=256, num_heads=8, mlp_dim=512,num_classes=9)

    try:
        train_model(model, train_loader, val_loader, device)
    except Exception as e:
        print(f"Training error: {e}")





