import os
import sys
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model_gcn_lstm import GCN_LSTM
from data_loader import load_kinematic_file
from video_utils import extract_motion_features_from_video

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = r"C:\Users\sahithi\Desktop\surgical-skill-assessment\results\gcn_lstm_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL -----------------
input_dim = 76
hidden_dim = 128
num_classes = 3  # Novice / Intermediate / Expert

model = GCN_LSTM(input_dim, hidden_dim, num_classes)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}, using random weights.")
model.to(device)
model.eval()

# ---------------- APP -------------------
app = Flask(__name__)
CORS(app)

# ---------------- HELPERS ----------------
def decode_label(label_id: int) -> str:
    return {0: "Novice", 1: "Intermediate", 2: "Expert"}.get(label_id, "Unknown")


def infer_task_from_motion(features: np.ndarray, filename: str = "") -> str:
    """
    Hybrid task inference:
    1️⃣ Uses filename hints if available.
    2️⃣ Falls back to motion-based heuristics if filename is ambiguous.
    """
    name = filename.lower()
    if "knot" in name:
        return "Knot Tying"
    if "needle" in name:
        return "Needle Passing"
    if "sutur" in name or "suture" in name:
        return "Suturing"

    # ---- fallback: motion-based inference ----
    try:
        mag_cols = features[:, 2::4]
        std_cols = features[:, 3::4]

        mean_mag = np.mean(mag_cols)
        std_mag = np.mean(std_cols)
        ratio = std_mag / (mean_mag + 1e-6)

        print(f"[Task Inference Fallback] mean={mean_mag:.3f}, std={std_mag:.3f}, ratio={ratio:.3f}")

        if mean_mag < 0.6 and ratio < 0.6:
            return "Knot Tying"
        elif (0.6 <= mean_mag <= 1.2 and 0.6 <= ratio <= 1.2):
            return "Suturing"
        else:
            return "Needle Passing"

    except Exception as e:
        print("⚠️ Task inference error:", e)
        return "Unknown"


def predict_skill_from_array(arr: np.ndarray):
    """Run model prediction on motion feature array."""
    with torch.no_grad():
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
        out = model(x)
        if isinstance(out, (tuple, list)):
            logits = next((o for o in out if isinstance(o, torch.Tensor) and o.ndim == 2), out[0])
        else:
            logits = out
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
    return pred, conf


# ---------------- ROUTES -----------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Surgical Skill Assessment (video+kinematic) API running."})


@app.route("/predict", methods=["POST"])
def predict_kinematic_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)

    data = load_kinematic_file(save_path)
    if data is None:
        return jsonify({"error": "Invalid kinematic file"}), 400

    pred, conf = predict_skill_from_array(data)
    return jsonify({
        "filename": filename,
        "task": "Kinematic File",
        "skill": decode_label(pred),
        "confidence": round(conf * 100, 2)
    })


@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)

    try:
        features = extract_motion_features_from_video(save_path, node_features=4, num_nodes=19)
        if features is None or features.size == 0:
            raise ValueError("Empty motion feature array.")
    except Exception as e:
        return jsonify({"error": "Failed to extract motion features", "detail": str(e)}), 500

    # ✅ Task and skill inference
    task = infer_task_from_motion(features, filename)
    label_id, conf = predict_skill_from_array(features)

    result = {
        "filename": filename,
        "task": task,
        "skill": decode_label(label_id),
        "confidence": round(conf * 100, 2)
    }

    return jsonify(result)


# ---------------- MAIN -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
