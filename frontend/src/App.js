import React from "react";
import FileUploadVideo from "./components/FileUploadVideo";
import "./styles.css";

function App() {
  return (
    <>
      {/* Background Video */}
      <video autoPlay loop muted playsInline className="video-bg">
        <source src="/bg.mp4" type="video/mp4" />
      </video>

      {/* Soft dark overlay */}
      <div className="overlay-dark"></div>

      <div className="App">
        <h1>🩺 Surgical Skill Assessment</h1>
        <h2>Upload video (.mp4 / .avi)</h2>
        <FileUploadVideo />
      </div>
    </>
  );
}

export default App;
