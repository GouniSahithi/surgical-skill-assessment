import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:5000",
});

export const uploadKinematicFile = async (formData) => {
  try {
    const response = await API.post("/predict", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  } catch (error) {
    console.error("❌ Prediction API Error:", error);
    throw error;
  }
};
