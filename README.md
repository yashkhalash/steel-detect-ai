# SteelSight AI — Tata Steel Quality Intelligence Platform

> **Tata Steel AI Hackathon 2026 Submission**  
> Steel surface defect detection with 5 industry-first intelligence modules

---

## 5 Unique Features (Not Found in Any Existing Tool)

| # | Feature | Why It's Unique |
|---|---------|----------------|
| 1 | **Defect Propagation Risk Forecaster** | Predicts how each defect evolves through Rolling → Annealing → Coating stages |
| 2 | **Real-Time Cost-of-Quality Calculator** | Converts defects to ₹ loss (scrap + rework + CO₂) per steel grade |
| 3 | **AI Shift Intelligence Report** | Agentic AI generates manager-ready shift narrative with corrective recommendations |
| 4 | **Root Cause Hypothesis Engine** | Maps each defect to ranked upstream process causes (metallurgy knowledge base) |
| 5 | **Mobile QC Inspector Mode** | Floor inspector captures coil photo on phone → instant PASS/FAIL → syncs to dashboard |

---

## Tech Stack

- **Backend**: Python 3.11 + Flask 3.0
- **AI Model**: YOLOv8n (fine-tuned on NEU-DET dataset, 6 defect classes)
- **XAI**: Grad-CAM heatmap overlays
- **Frontend**: Vanilla JS + Chart.js (no framework — pure speed)
- **Dataset**: NEU Surface Defect Database (Northeastern University)

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000          → Main inspection page
http://localhost:5000/dashboard → Shift intelligence dashboard
http://localhost:5000/mobile   → Mobile QC inspector
```

---

## Training the Model (Optional — app runs in simulation mode without this)

### Step 1: Download NEU-DET Dataset
Download from: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

### Step 2: Install ultralytics
```bash
pip install ultralytics torch torchvision
```

### Step 3: Train
```bash
python model/train.py --neu-src /path/to/NEU-DET --epochs 80 --model n
```

Best weights saved to `model/best.pt`

### Step 4: Enable real inference
In `app.py`, replace `simulate_detection()` with:
```python
from model.inference import real_detect
dets = real_detect(upath, conf_threshold=0.45)
```

---

## Defect Classes (NEU-DET)

| Class | Severity | Primary Cause |
|-------|----------|---------------|
| Crazing | Medium | Rapid thermal cycling |
| Inclusion | High | Slag carryover, poor argon stirring |
| Patches | Low | Scale build-up on furnace rolls |
| Pitted Surface | High | High dissolved hydrogen (>2 ppm) |
| Rolled-in Scale | Medium | Descaler nozzle blockage |
| Scratches | Low | Guide groove wear, handling damage |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/inspect` | Upload image → returns full inspection JSON |
| GET | `/dashboard` | Shift intelligence dashboard UI |
| GET | `/mobile` | Mobile QC inspector UI |
| GET | `/api/dashboard-data` | Dashboard JSON (verdicts, defect freq, timeline) |
| GET | `/api/shift-report?n=50` | AI-generated shift narrative |
| GET | `/api/seed-demo` | Load 15 demo inspections |
| GET | `/api/grades` | List steel grades with cost config |

---

## Project Structure

```
steel-defect-app/
├── app.py                  # Flask routes + 5 unique intelligence engines
├── model/
│   ├── train.py            # YOLOv8 fine-tuning on NEU-DET
│   └── inference.py        # Production detection + Grad-CAM
├── templates/
│   ├── index.html          # Main inspection page (dark industrial UI)
│   ├── dashboard.html      # Shift intelligence dashboard + charts
│   └── mobile.html         # Mobile QC inspector (responsive)
├── static/
│   └── uploads/            # Processed images
├── requirements.txt
└── README.md
```

---

## About

Built for the **Tata Steel AI Hackathon 2026** — demonstrating how AI can move from defect detection to full decision intelligence: knowing *what* the defect is, *why* it occurred, *where* it will propagate, and *how much* it costs in real time.

This mirrors Tata Steel's bimodal AI strategy: **Narrow AI** (defect-specific models) combined with **Agentic AI** (shift reports, mobile deployment, cost reasoning).
# steel-detect-ai
