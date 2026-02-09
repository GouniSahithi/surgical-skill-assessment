// src/components/FileUploadVideo.jsx
import React, { useState } from "react";

export default function FileUploadVideo() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const onFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const upload = async () => {
  if (!file) return alert("Choose a video first.");
  const fd = new FormData();
  fd.append("file", file);

  setLoading(true);
  try {
    const res = await fetch("http://127.0.0.1:5000/predict_video", {
      method: "POST",
      body: fd,
    });
    const json = await res.json();
    console.log("✅ Response from backend:", json); // 👈 ADD THIS
    setResult(json);
  } catch (err) {
    console.error(err);
    alert("Upload failed: " + err.message);
  } finally {
    setLoading(false);
  }
};


  return (
    <div style={{ padding: 20 }}>
      <h2>Upload video (.mp4 / .avi)</h2>
      <input type="file" accept="video/*" onChange={onFileChange} />
      <div style={{ marginTop: 10 }}>
        <button onClick={upload} disabled={!file || loading}>
          {loading ? "Processing..." : "Upload & Predict"}
        </button>
      </div>

     {result && (
  <div className="result-card">
    <h3><b>Filename:</b> {result.filename}</h3>
    <p><b>Task:</b> {result.task}</p>
    <p><b>Predicted Skill:</b> {result.skill}</p>
    <p><b>Confidence:</b> {result.confidence}%</p>
  </div>
)}


    </div>
  );
}
