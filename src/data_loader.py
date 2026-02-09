import os
import numpy as np
import torch

# ==================================================
# 1️⃣ SUBJECT → SKILL MAPPING (based on official JIGSAWS paper)
# ==================================================
# Skill levels:
# 0 → Novice (<10 hours)
# 1 → Intermediate (10–100 hours)
# 2 → Expert (>100 hours)

SKILL_MAP = {
    "B": 0,  # novice
    "G": 0,  # novice
    "H": 0,  # novice
    "I": 0,  # novice
    "C": 1,  # intermediate
    "F": 1,  # intermediate
    "D": 2,  # expert
    "E": 2,  # expert
}

# ==================================================
# 2️⃣ LOAD SINGLE KINEMATIC FILE
# ==================================================
def load_kinematic_file(file_path):
    """Load a single .txt kinematic file and return np.array."""
    try:
        data = np.loadtxt(file_path, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print(f"⚠️ Skipped {os.path.basename(file_path)}: {e}")
        return None


# ==================================================
# 3️⃣ MAIN DATA LOADER
# ==================================================
def load_jigsaws_data(data_path):
    """
    Reads all JIGSAWS task folders and returns (X, y) tensors.
    Expected structure:
      data_path/
        ├── Knot_Tying/kinematic_allgestures/AllGestures/*.txt
        ├── Needle_Passing/kinematic_allgestures/AllGestures/*.txt
        └── Suturing/kinematic_allgestures/AllGestures/*.txt
    """
    all_data, all_labels = [], []

    # Loop through each task directory
    for task_name in ["Knot_Tying", "Needle_Passing", "Suturing"]:
        task_dir = os.path.join(data_path, task_name, "kinematic_allgestures", "AllGestures")

        if not os.path.exists(task_dir):
            print(f"⚠️ Skipping {task_name} (folder not found)")
            continue

        print(f"📂 Loading {task_name} data from {task_dir}")

        for fname in os.listdir(task_dir):
            if not fname.endswith(".txt"):
                continue

            file_path = os.path.join(task_dir, fname)
            parts = fname.split("_")
            if len(parts) < 2:
                print(f"⚠️ Skipping {fname} (unexpected file name format)")
                continue

            subj_id = parts[1][0].upper()  # e.g., 'B001' → 'B'

            if subj_id not in SKILL_MAP:
                print(f"⚠️ Skipping {fname} (unknown subject ID: {subj_id})")
                continue

            data = load_kinematic_file(file_path)
            if data is None or data.size == 0:
                continue

            label = SKILL_MAP[subj_id]
            all_data.append(data)
            all_labels.append(label)

            print(f"✅ Loaded {fname} → shape {data.shape}, label={label}")

    if not all_data:
        raise ValueError("❌ No valid data found — check your data_path or folder structure!")

    # ==================================================
    # 4️⃣ NORMALIZE SEQUENCE LENGTHS
    # ==================================================
    seq_lengths = [d.shape[0] for d in all_data]
    median_len = int(np.median(seq_lengths))
    feature_dim = all_data[0].shape[1]

    print(f"📏 Normalizing all sequences to {median_len} frames")

    fixed_data = []
    for d in all_data:
        if d.shape[0] > median_len:
            d = d[:median_len, :]
        elif d.shape[0] < median_len:
            pad = np.zeros((median_len - d.shape[0], feature_dim), dtype=np.float32)
            d = np.vstack([d, pad])
        fixed_data.append(d)

    X = np.stack(fixed_data, axis=0)
    y = np.array(all_labels)

    print(f"✅ Final dataset: {len(X)} samples | Shape = {X.shape}")
    print(f"📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ==================================================
    # 5️⃣ SAVE PROCESSED DATA (optional cache)
    # ==================================================
    save_dir = os.path.join(data_path, "processed")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)
    print(f"💾 Saved preprocessed data to {save_dir}")

    # ==================================================
    # 6️⃣ RETURN PYTORCH TENSORS
    # ==================================================
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor, y_tensor
