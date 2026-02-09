 Surgical Skill Assessment using GCN-LSTM

Project Overview:

This project implements a machine learning–based surgical skill assessment system using graph-based deep learning techniques.
It analyzes laparoscopic surgical task data and predicts the skill level of a surgeon based on motion patterns and kinematic features.

The system leverages a Graph Convolutional Network (GCN) to model spatial relationships between surgical tool features and an LSTM network to capture temporal dynamics. A Flask-based backend handles inference, while a frontend interface enables users to upload surgical videos and view predictions.

The project is designed for applications in medical training, surgical education, and skill evaluation.

Features:

Surgical Skill Prediction: Classifies surgical performance into skill levels

Graph-Based Learning: Uses GCN to model relationships between tool features

Temporal Modeling: LSTM captures time-dependent motion patterns

Video/Kinematic Data Support: Processes laparoscopic task data

Web Interface: Upload videos and receive predictions via UI

Confidence Scores: Displays prediction confidence for each skill level

Model Architecture:

Graph Convolutional Network (GCN)

Captures spatial dependencies among kinematic features

Long Short-Term Memory (LSTM)

Models temporal evolution of surgical motion

Softmax Output Layer

Produces probability distribution over skill levels

Repository Structure:
surgical-skill-assessment/
│
├── backend/                     # Flask backend
│   ├── app.py
│   ├── app_video.py
│   ├── inference_utils.py
│   ├── video_utils.py
│   └── requirements.txt
│
├── frontend/                    # Frontend UI
│   ├── public/
│   └── src/
│
├── src/                         # ML training & evaluation code
│   ├── model_gcn_lstm.py
│   ├── train.py
│   ├── evaluate.py
│   ├── data_loader.py
│   └── utils.py
│
├── data/                        # Dataset directory (not included)
├── results/                     # Trained models (ignored in Git)
├── README.md
└── .gitignore

Dataset:

This project uses the JIGSAWS Surgical Skill Assessment Dataset.

 The dataset is not included in this repository due to size and licensing constraints.

You can request access from the official source:
🔗 http://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/

 Setup Instructions
🔹 Prerequisites

Python 3.8 or higher

Git

🔹 Clone the Repository
git clone https://github.com/GouniSahithi/surgical-skill-assessment.git
cd surgical-skill-assessment

🔹 Setting Up a Virtual Environment (Recommended)

On Windows:

python -m venv venv
venv\Scripts\activate


On macOS / Linux:

python3 -m venv venv
source venv/bin/activate

🔹 Installing Dependencies
pip install -r backend/requirements.txt


This installs required packages including:

Flask

PyTorch

NumPy

OpenCV

scikit-learn

▶️ Running the Application
Start the Backend Server
cd backend
python app.py


If successful, you will see:

{ "message": "Surgical Skill Assessment API is running!" }

Access the Frontend

Open your browser and go to:

http://localhost:5000


Upload a surgical video

Select task type

Click Predict

View predicted skill level and confidence score

📈 Model Training

To train the model from scratch:

Place the JIGSAWS dataset in the data/ directory

Configure paths in src/train.py

Run:

python src/train.py


⚠️ GPU training is recommended (Google Colab preferred) for faster convergence.

🛑 Notes

Large files (datasets, videos, model weights) are excluded using .gitignore

Training on CPU may be slow

Predictions depend on sufficient training epochs and balanced data

👩‍💻 Author

Sahithi Gouni
Computer Science Engineering Student
Interests: AI/ML, Computer Vision, Healthcare AI

🔮 Future Enhancements

Improve skill classification accuracy

Add real-time video inference

Deploy full application on cloud

Add explainability and visual feedback

Integrate surgeon performance analytics