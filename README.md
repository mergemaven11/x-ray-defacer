# 🩻 X-Ray Defacer (Roboflow Tech Challenge)

A **FastAPI + Roboflow** web application that automatically detects and anonymizes identifiable facial regions (eyes, nose, mouth) in medical X-rays.  
The app demonstrates a complete Roboflow workflow: **data labeling, model training, assisted annotation, deployment, and API integration**.

Built and maintained by **Tobias Scott**.

---

## 🚀 Overview

This project uses **Roboflow’s Object Detection API** to find and mask patient-identifying features in X-ray scans.  
Once uploaded, each image is sent through a **Roboflow-hosted model** for inference.  
Detections are automatically **black-boxed (or blurred)** to anonymize the scan before display or download.

---

## 🧠 Roboflow Integration Workflow

| Stage | Tool / Feature | Purpose |
|-------|----------------|----------|
| **Data Labeling** | Roboflow Annotate | Labeled ~400 X-rays with `eyes`, `nose`, `mouth` |
| **Model Training** | Roboflow Train | Iteratively trained v8–v10 with augmentations (flip, rotation, noise) |
| **Label Assist** | Model-Assisted Labeling | Bootstrapped labeling on unlabeled images |
| **Evaluation** | Roboflow Metrics | mAP: 14.8%, Precision: 15.3%, Recall: 76.0% (improved with v10) |
| **Deployment** | Roboflow Hosted Inference | Used REST API endpoint for server-side detection |
| **Automation** | Python + FastAPI | Applied black-box anonymization per bounding box |

---

## 🖥️ Tech Stack

| Layer | Technology |
|-------|-------------|
| **Backend API** | FastAPI (Python 3.11) |
| **ML Inference** | Roboflow Hosted Object Detection API |
| **Frontend** | Jinja2 + Tailwind-inspired custom CSS |
| **Containerization** | Docker + Docker Compose |
| **Hosting** | Railway (backend) + Netlify (frontend) |
| **Image Processing** | Pillow (PIL) |
| **Environment Management** | python-dotenv |

---

## 📂 Project Structure

```
xray-defacer/
├── app.py                  # FastAPI backend
├── Dockerfile              # Railway container build
├── docker-compose.yml      # Local dev configuration
├── .env                    # Local-only Roboflow credentials
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   └── style.css           # Custom CSS styling
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Environment Variables

For local development, create a `.env` file:

```bash
MODEL_URL=https://detect.roboflow.com/xray-defacer-v10/10
ROBOFLOW_API_KEY=rf_your_api_key_here
CONFIDENCE_THRESHOLD=0.34
SAVE_OUTPUTS=true
```

These variables are injected at runtime by `python-dotenv` or directly in Railway’s environment settings.

---

## 🧩 Local Development

1. **Clone the repo**
   ```bash
   git clone https://github.com/mergemaven11/xray-defacer.git
   cd xray-defacer
   ```

2. **Create `.env`**
   ```bash
   MODEL_URL=...
   ROBOFLOW_API_KEY=...
   ```

3. **Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. Open [http://localhost:8000](http://localhost:8000)

---

## 🧠 How It Works

1. The user uploads an X-ray image.  
2. The backend calls the **Roboflow Hosted Model** via REST API:
   ```python
   response = requests.post(
       f"{MODEL_URL}?api_key={ROBOFLOW_API_KEY}",
       files={"file": ("xray.jpg", image_bytes, "image/jpeg")},
   )
   ```
3. Predictions (x, y, width, height, class, confidence) are parsed.  
4. Each detection is **black-boxed** or blurred using Pillow:
   ```python
   draw.rectangle([left, top, right, bottom], fill="black")
   ```
5. The app renders before/after views and metadata (detections, confidence, latency).  
6. Users can also test the `/api/deface` endpoint to see raw JSON inference results.

---


### 🔹 CORS Configuration
Add to `app.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://xray-defacer.netlify.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🧾 API Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| `GET` | `/` | Render upload form |
| `POST` | `/deface` | Upload image → get HTML with before/after |
| `POST` | `/api/deface` | Upload image → get JSON predictions only |

Example JSON:
```json
{
  "num_predictions": 3,
  "predictions": [
    { "class": "eye", "confidence": 0.91 },
    { "class": "nose", "confidence": 0.88 },
    { "class": "mouth", "confidence": 0.85 }
  ],
  "inference_time": 0.62
}
```

---

## 🎨 Frontend Features

- Elegant gradient UI inspired by Tailwind.
- Dynamic file name and preview.
- Loading spinner overlay during inference.
- Dual mode: **Anonymize (image)** or **Get JSON Output (API)**.
- Model metrics panel (mAP, precision, recall).
- Download button for defaced image.

---

## 🧠 Lessons Learned

- **Data quality** and **label diversity** drastically impact model recall.  
- Label Assist speeds up scaling annotation but requires manual review.  
- Confidence threshold tuning (via Roboflow charts) improves anonymization accuracy.  
- Roboflow’s REST API enables flexible deployment — perfect for lightweight privacy tools.

---

## 🏗️ Future Improvements

- Add **confidence heatmaps** for visual debugging.  
- Integrate **Roboflow Workflow** for real-time batch inference.  
- Add **OAuth2** authentication for organizational dashboards.  
- Include **OpenAI-assisted auto-labeling** experiments.

---

## 🧾 License

MIT © 2025 Tobias Scott  
Made with ❤️ and Roboflow.
