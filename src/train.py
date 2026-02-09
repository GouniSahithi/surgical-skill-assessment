import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from model_gcn_lstm import GCN_LSTM
from data_loader import load_jigsaws_data

# ==================================================
# CONFIG
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = r"C:\Users\sahithi\Desktop\surgical-skill-assessment\data\JIGSAWS"
num_epochs = 20
batch_size = 8
learning_rate = 0.001

print(f"✅ Using device: {device}")

# ==================================================
# LOAD DATA
# ==================================================
X, y = load_jigsaws_data(data_path)
print(f"📥 Loaded dataset: X={X.shape}, y={y.shape}")

dataset = TensorDataset(X, y)

# Split data
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ==================================================
# MODEL SETUP
# ==================================================
input_dim = X.shape[2]  # 76
hidden_dim = 128
num_classes = 3

model = GCN_LSTM(input_dim, hidden_dim, num_classes, dropout=0.3).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
criterion = nn.CrossEntropyLoss()

# ==================================================
# TRAINING LOOP
# ==================================================
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    scheduler.step(epoch_loss)

    print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# ==================================================
# EVALUATION
# ==================================================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"🎯 Final Test Accuracy: {test_acc:.2f}%")

# ==================================================
# SAVE MODEL
# ==================================================
torch.save(model.state_dict(), "../results/gcn_lstm_model.pth")
print("💾 Model saved successfully!")
