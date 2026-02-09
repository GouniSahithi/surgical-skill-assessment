import torch
import os

# ====== PATHS ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, "data", "JIGSAWS")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ====== TASKS & SUBJECTS ======
TASKS = ["Suturing", "Knot_Tying", "Needle_Passing"]
SUBJECTS = [f"{s}{i:03d}" for s in "BCDEFGHI" for i in range(1, 6)]

# ====== TRAINING PARAMETERS ======
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 20
NUM_CLASSES = 15
NUM_NODES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Using device: {DEVICE}")
