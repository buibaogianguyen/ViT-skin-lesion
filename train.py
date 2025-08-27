import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.to(device)

    