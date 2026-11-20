# SIGNBRIDGE

**SIGNBRIDGE** is a local, camera-based Indian Sign Language (ISL) static-sign recognition prototype. It uses MediaPipe Hands to detect hand landmarks, a trained scikit-learn classifier to predict supported labels, a FastAPI backend, and a React + Vite frontend.

> **Prototype notice:** This is an educational demonstration for a small set of static signs. It is not a complete ISL translator, does not interpret signed sentences or motion-based signs, and must not be presented as a certified accessibility or communication system. Predictions should be checked against reliable ISL references.

## Features

- Live webcam interface in the browser
- MediaPipe hand landmark detection
- FastAPI endpoint for image-frame recognition
- Random Forest classifier loaded from `models/sign_model.joblib`
- Dataset stored as landmark rows in `dataset/sign_landmarks.csv`
- Separate React frontend and Python backend

## Project structure

```text
SIGNBRIDGE/
├── backend/
│   ├── main.py
│   ├── recognition.py
│   ├── collect_data.py
│   ├── collect_data_backup.py
│   ├── hand_tracker.py
│   └── train_model.py
├── dataset/
│   └── sign_landmarks.csv
├── models/
│   └── sign_model.joblib
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── package-lock.json
├── requirements.txt
└── README.md
```

## Run SIGNBRIDGE on your exhibition laptop

**Important:** Run both the backend and frontend on the same laptop. The browser frontend sends camera frames to the local FastAPI server at `http://127.0.0.1:8000`. The app is not a public website and should be demonstrated locally.

### 1. Install the prerequisites (before exhibition day)

Install:

- **Python 3.12** (64-bit recommended)
- **Node.js 20.19+ or 22.12+** (includes npm)
- A modern version of Chrome or Edge
- A working webcam

Check the installations in PowerShell:

```powershell
python --version
node --version
npm --version
```

### 2. Download the project

On the exhibition laptop, open the repository:

[https://github.com/farixhere/SIGNBRIDGE](https://github.com/farixhere/SIGNBRIDGE)

Choose **Code → Download ZIP**, extract it, then open the extracted `SIGNBRIDGE` folder.

Alternatively, if Git is installed:

```powershell
git clone https://github.com/farixhere/SIGNBRIDGE.git
cd SIGNBRIDGE
```

### 3. Set up the Python backend

Open **PowerShell window 1** and move into the project folder. If the folder is on your Desktop, for example:

```powershell
cd "$HOME\Desktop\SIGNBRIDGE"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in that PowerShell window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the backend packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the API:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Keep this window open. Confirm that the API is running by opening:

- Health check: http://127.0.0.1:8000/health
- API documentation: http://127.0.0.1:8000/docs

The health response should show `"status": "healthy"`. Check `"model_loaded": true` to confirm that the classifier file was loaded.

### 4. Start the React frontend

Open **PowerShell window 2**. Go to the frontend folder:

```powershell
cd "$HOME\Desktop\SIGNBRIDGE\frontend"
```

Install frontend packages (do this before the exhibition):

```powershell
npm install
```

Start the website:

```powershell
npm run dev -- --host 127.0.0.1
```

Keep this window open. Vite will show a local address, usually:

http://127.0.0.1:5173/

Open that address in Chrome or Edge.

### 5. Demonstrate it

1. Keep both PowerShell windows running.
2. Open the frontend at **http://127.0.0.1:5173/**.
3. Allow camera permission when the browser asks.
4. Keep your hand clearly visible, well-lit, and inside the camera frame.
5. Try one of the signs supported by the model.
6. Wait for the recognition result. If the hand is detected but the result is uncertain, show the sign again with a clear hand position.
7. Explain that this is a static-sign prototype and not full sentence translation.

**Do not close either terminal while presenting.** The backend must remain running for recognition, and the Vite frontend must remain running for the interface.

## Quick start checklist for exhibition day

- [ ] Laptop plugged in and charged
- [ ] Webcam tested
- [ ] Python and Node.js installed
- [ ] Dependencies installed before the event
- [ ] Internet not required once dependencies are installed and the project is on the laptop
- [ ] Backend terminal running on port 8000
- [ ] Frontend terminal running on port 5173
- [ ] Browser camera permission allowed
- [ ] Good lighting and uncluttered camera background

## Troubleshooting

### The website opens, but the API is offline

Check that PowerShell window 1 is still running. Open http://127.0.0.1:8000/health. If it does not load, return to the project root, activate `.venv`, and start the backend command again.

### Camera permission is denied

Use `http://127.0.0.1:5173/` on the same laptop, allow camera access in the browser's site settings, and make sure another application is not using the webcam.

### The API says the model is not loaded

Confirm that `models/sign_model.joblib` exists. The backend loads this file when it starts. Restart the backend after replacing or retraining the model.

### A sign is detected but no prediction appears

The classifier intentionally avoids presenting low-confidence predictions as recognized signs. Improve lighting, show the hand fully, and try again. A hand-detection result is not the same as a correct sign prediction.

### Dependency installation fails

Use Python 3.12 (64-bit) and a supported Node.js release. Run the install commands from the correct folders. A working internet connection is needed for the first package installation; after that, the local demonstration can run offline.

## Dataset and model notes

The training script expects **at least two different labels** and **at least 8 samples for every label**. It stops if those conditions are not met. To retrain after collecting enough verified examples:

```powershell
# From the project root, with .venv activated
python -m backend.train_model
```

After successful training, restart the backend so it loads the new model. Do not treat a high score from a small dataset as proof of real-world accuracy.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Basic API status |
| GET | `/health` | Backend and model health |
| GET | `/api/signs` | Supported classifier labels |
| POST | `/api/recognize` | Recognize a sign from an uploaded image frame |

Interactive API documentation is available at http://127.0.0.1:8000/docs while the backend is running.

## Tech stack

- React 19 and Vite
- Python, FastAPI and Uvicorn
- OpenCV and MediaPipe
- NumPy, scikit-learn and joblib

## Responsible-use note

SIGNBRIDGE is a student prototype. ISL is a full language with grammar, regional variation, facial expressions, and movement. This project currently classifies a limited set of static hand poses only. It should not replace a qualified interpreter or be relied on for important communication.
