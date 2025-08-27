import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.dataloader import HAM10000
from data.data_manager import load_metadata, init_db


def train_model(model, train_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.to(device)

    for epoch in range(epochs):
        model.train()

        train_loss = 0
        
        for (images, labels) in range(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Epoch {epoch+1}\nAvg Train Loss: {train_loss/len(train_loader):.4f}')
    
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(224,224),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 32

    session = init_db()

    dataset_path = load_metadata(session)
    dataset = HAM10000(dataset_path=dataset_path, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size, num_workers=2)



