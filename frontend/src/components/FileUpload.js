import React, { useState } from "react";

function FileUpload() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log("API Response:", data); // 👀 debug

      setResult(data);
    } catch (error) {
      console.error("Upload error:", error);
      alert("Error connecting to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <h1>🩺 Surgical Skill Assessment</h1>
      <p>Upload a kinematic file (.txt) to classify the surgeon’s skill level</p>

      <div className="upload-box">
        <input type="file" accept=".txt" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={loading}>
          {loading ? "Processing..." : "Upload & Predict"}
        </button>
      </div>

      {result && (
        <div className="result-box">
          <h2>Prediction Result</h2>
          <p><strong>Filename:</strong> {result.filename}</p>
          <p><strong>Task:</strong> {result.task}</p>
          <p><strong>Predicted Skill:</strong> {result.skill}</p>
          <p><strong>Confidence:</strong> {result.confidence.toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}

export default FileUpload;
