from simple_cnn import Simple2DConvNN
import torch
from torch.utils.data import DataLoader
from datasets_utils import get_datasets
from utils import get_device

num_epochs = 10
train_batch_size = 4

device = get_device()
train_dataset, val_dataset, _ = get_datasets()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=train_batch_size * 2, shuffle=False)

model = Simple2DConvNN().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(f"Epoch {epoch} | Train loss: {loss}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Average train loss: {avg_loss}")

    with torch.no_grad():
        total_val_loss = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
        
        print(f"Epoch {epoch} | Average val loss: {total_val_loss/len(val_loader)}")

