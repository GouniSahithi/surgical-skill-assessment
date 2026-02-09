import torch
from config import *
from data_loader import JIGSAWSDataset
from model_gcn_lstm import GCN_LSTM_Model, create_adjacency

dataset = JIGSAWSDataset(DATA_ROOT, TASKS, SUBJECTS)
model = GCN_LSTM_Model(num_nodes=NUM_NODES, in_node_features=4, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/model_final.pt", map_location=DEVICE))
model.eval()

adj = create_adjacency(NUM_NODES).to(DEVICE)

correct, total = 0, 0
with torch.no_grad():
    for X, y in dataset:
        X = X.view(1, X.shape[0], NUM_NODES, 4).to(DEVICE)
        y = y.to(DEVICE)
        logits, _ = model(X, adj)
        preds = logits.argmax(-1).squeeze()
        correct += (preds == y).sum().item()
        total += y.numel()

acc = 100 * correct / total
print(f"✅ Evaluation Accuracy: {acc:.2f}%")
