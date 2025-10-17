# 🚦 Smart Traffic Management System

## 📌 Project Overview
The **Smart Traffic Management System** is a basic working prototype designed to demonstrate how artificial intelligence can be used to **control traffic signal timing dynamically** based on the amount of traffic in each lane.

This project uses **YOLOv8 (You Only Look Once)** for vehicle detection and **OpenCV** for video processing.  
It is **not a high-end system**, but a **proof-of-concept demonstration** showing how such technology can be implemented in real-world traffic control systems.

---

## 🎯 Objective
The main goal of this project is to **reduce unnecessary waiting time** at traffic signals by increasing or decreasing the green light duration based on the number of vehicles detected in a single lane.

---

## 🧠 How It Works
1. **Video Feed Input:** A video of a traffic lane (from a webcam or file) is analyzed frame-by-frame.  
2. **Vehicle Detection:** YOLOv8 detects and counts vehicles in real time.  
3. **Decision Logic:** Based on the vehicle count, the system decides whether to **increase or decrease the signal duration**.  
4. **Display Result:** The updated signal timing is displayed as output for demonstration.

---

## 🛠️ Technologies Used
- **Python**
- **YOLOv8 (Ultralytics)**
- **OpenCV (cv2)**
- **NumPy**

---

## 🧩 Features
- Real-time vehicle detection using YOLOv8  
- Dynamic signal control based on traffic density  
- Works on pre-recorded traffic videos  
- Simple, educational, and easy-to-understand implementation  

---

## 📂 Project Structure
smart_traffic_system/
│
├── main.py # Main Python script
├── traffic.mp4 # Sample traffic video
├── yolov8n.pt # YOLOv8 pre-trained model
└── README.md # Project documentation
---

## ⚙️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Hari0218/smart_traffic_system.git

for dependencies install
pip install ultralytics opencv-python numpy
run the script 
python main.py

