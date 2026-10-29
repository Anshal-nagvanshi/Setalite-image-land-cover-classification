# Satellite Land Cover Classification System (MERN Stack)

Full-stack MERN (MongoDB, Express, React, Node.js) implementation based on the deep learning Sentinel-2 & EuroSAT satellite land cover classification dashboard.

## 📁 Project Architecture

```
setalite land cover classification/
├── client/                     # Frontend (React 18, Vite, Tailwind CSS, Chart.js)
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx             # React dashboard & navigation logic
│       └── index.css           # Tailwind + Custom animation styles
│
├── server/                     # Backend (Node.js, Express, Multer, Mongoose)
│   ├── package.json
│   ├── server.js               # REST API endpoints & classification simulator
│   ├── .env                    # PORT & MongoDB URI
│   └── models/
│       ├── Classification.js   # Mongoose model for patch classifications
│       ├── Project.js          # Mongoose model for batch experiments
│       └── Setting.js          # Mongoose model for system configs & profile
│
└── gemini-code-1793197765572.html  # Original HTML reference
```

---

## 🚀 How to Run the Application

### 1. Start the Backend API (Server)

In your terminal:
```bash
cd server
npm install
npm start
```
- The backend will start at: `http://localhost:5000`
- It connects to MongoDB (`mongodb://127.0.0.1:27017/satellite_classification`). If MongoDB is not active on your machine, it automatically uses an **in-memory resilient fallback** so everything works out of the box!

### 2. Start the Frontend (Client)

In a separate terminal:
```bash
cd client
npm install
npm run dev
```
- Open `http://localhost:5173` in your browser.

---

## 🛰 Features

1. **Dashboard & Deep Learning Inference**:
   - Drag & drop Sentinel-2 satellite patches or upload custom JPG/PNG images up to 10MB (handled via Multer).
   - Test preset EuroSAT benchmark samples (Forest, Water, Urban, Cropland).
   - Real-time deep learning feature extraction simulation and EuroSAT class confidence distribution rendered via Chart.js.
   - Saves classification history and spectral prediction results.

2. **Projects & Batch Datasets**:
   - Create, list, duplicate, and archive satellite classification datasets and batches.
   - Fully synced with the database.

3. **System Settings**:
   - Switch DL architectures (ResNet-50, EfficientNet-B0, Custom CNN).
   - Configure image normalization (Min-Max vs. Z-Score).
   - Test-Time Augmentation (TTA) toggles.

4. **Account & Profile**:
   - Editable researcher profile (Full Name, Username, Email) synced to backend.
