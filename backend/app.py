import os
import sys
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add src folder to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
SRC_DIR = os.path.join(PARENT_DIR, "src")
sys.path.append(SRC_DIR)

from model_gcn_lstm import GCN_LSTM
from data_loader import load_kinematic_file

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = r"C:\Users\sahithi\Desktop\surgical-skill-assessment\results\gcn_lstm_model.pth"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# MODEL SETUP
# ------------------------------------------------------------
input_dim = 76
hidden_dim = 128
num_classes = 3  # novice / intermediate / expert

model = GCN_LSTM(input_dim, hidden_dim, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ------------------------------------------------------------
# FLASK APP
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def get_task_type(filename: str) -> str:
    """Infer surgical task from filename"""
    name = filename.lower()
    if "knot" in name:
        return "Knot Tying"
    elif "needle" in name:
        return "Needle Passing"
    elif "suturing" in name:
        return "Suturing"
    return "Unknown"


def predict_skill(data: np.ndarray):
    """Run model prediction on kinematic data"""
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        out = model(x)

        # --- handle unexpected shapes ---
        if out.ndim == 3:  # [batch, seq_len, num_classes]
            out = out.mean(dim=1)
        elif out.ndim == 1:
            out = out.unsqueeze(0)

        # --- clamp values to avoid overflow ---
        out = torch.clamp(out, -50, 50)

        # --- compute softmax safely ---
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]

        # --- ensure valid probabilities ---
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.array([1/3, 1/3, 1/3])

        pred = int(np.argmax(probs))
        conf = float(np.max(probs))

    return pred, conf


def decode_label(label_id: int) -> str:
    return {0: "Novice", 1: "Intermediate", 2: "Expert"}.get(label_id, "Unknown")


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Surgical Skill Assessment API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    data = load_kinematic_file(save_path)
    if data is None:
        return jsonify({"error": "Invalid file format"}), 400

    # Predict
    label_id, conf = predict_skill(data)
    skill = decode_label(label_id)
    task = get_task_type(filename)

    result = {
        "task": task,
        "skill": skill,
        "confidence": round(conf * 100, 2),  # 0–100%
        "filename": filename,
    }

    return jsonify(result)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
