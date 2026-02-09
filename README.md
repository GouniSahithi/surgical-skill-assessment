# **Surgical Skill Assessment**



Project Type: Machine Learning | Healthcare AI

Domain: Surgical Skill Evaluation, Computer Vision, Graph Neural Networks.



### 1\. Project Overview



* This project implements a Surgical Skill Assessment system using deep learning techniques to evaluate a surgeon’s skill level from laparoscopic task data.
* The system leverages Graph Convolutional Networks (GCN) to capture spatial relationships between surgical tool features and Long Short-Term Memory (LSTM) networks to model temporal motion patterns.



* A Flask-based backend handles inference, while a web-based frontend allows users to upload surgical videos and receive skill predictions with confidence scores.



* This project is designed to support medical training, surgical education, and objective skill evaluation.



### 2\. Features



* Surgical Skill Prediction: Classifies surgeon performance into skill levels



* Graph-Based Learning: Uses GCN to model relationships between tool kinematics



* Temporal Analysis: LSTM captures time-dependent motion patterns



* Video Upload Support: Accepts laparoscopic surgical videos



* Confidence Scores: Displays prediction confidence for each skill class



* Web Interface: Simple and user-friendly frontend



### 3\. Model Architecture



* Graph Convolutional Network (GCN): Models spatial dependencies among surgical tool features



* Long Short-Term Memory (LSTM): Captures temporal dynamics of surgical motion



* Softmax Classifier: Outputs probability distribution over skill levels



### 4\. Repository Structure



surgical-skill-assessment/

│

├── backend/                     # Flask backend

│   ├── app.py                   # Main backend server

│   ├── app\_video.py             # Video-based inference

│   ├── inference\_utils.py       # Prediction utilities

│   ├── video\_utils.py           # Video preprocessing

│   └── requirements.txt         # Backend dependencies

│

├── frontend/                    # Frontend UI

│   ├── public/

│   └── src/

│

├── src/                         # ML training \& evaluation

│   ├── model\_gcn\_lstm.py        # GCN-LSTM model

│   ├── train.py                 # Training script

│   ├── evaluate.py              # Evaluation script

│   ├── data\_loader.py           # Data loading utilities

│   └── utils.py                 # Helper functions

│

├── data/                        # Dataset directory (not included)

├── results/                     # Model outputs (ignored in Git)

├── README.md                    # Project documentation

└── .gitignore



### 5.Dataset



This project uses the JIGSAWS Surgical Skill Assessment Dataset.



1. The dataset is not included in this repository due to size and license restrictions.



Dataset source:

🔗 http://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws\_release/



### 6.Setup Instructions



1. Prerequisites



* Python 3.8 or higher



* Git



2\.	Clone the Repository



git clone https://github.com/GouniSahithi/surgical-skill-assessment.git

cd surgical-skill-assessment



3\.	Create a Virtual Environment



Windows:



python -m venv venv

venv\\Scripts\\activate



macOS / Linux:



python3 -m venv venv

source venv/bin/activate



4\.	Install Dependencies



pip install -r backend/requirements.txt



### 7.Running the Application



1. Start Backend Server:



cd backend

python app.py



If successful, you will see:

{ "message": "Surgical Skill Assessment API is running!" }



2\.	 Access the Frontend



* Open a browser and navigate to:



* http://localhost:5000



* Upload a surgical video



* Select the task



* Click Predict



* View predicted skill level and confidence score



### 8.Model Training



To train the model from scratch:



* Place the JIGSAWS dataset inside the data/ directory



* Configure dataset paths in src/train.py



* Run the training script:



&nbsp;           .python src/train.py





\*\*GPU training (Google Colab) is recommended for faster convergence.



### 9.Notes



* Large files (datasets, videos, trained models) are excluded using .gitignore



* Training on CPU may be slow



* Model performance improves with sufficient training epochs and balanced data



### 10.Author



1. Sahithi Gouni
   
2. Computer Science Engineering Student
   
3. Interests: AI/ML, Computer Vision, Healthcare AI



### 11.Future Enhancements



* Improve skill classification accuracy



* Add real-time video inference



* Deploy full-stack application on cloud



* Add explainability and performance analytics





