# InfraScan – AI-Powered Crack & Damage Assessment for Civil Infrastructure

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-green.svg)
![React](https://img.shields.io/badge/frontend-React-cyan.svg)
![PostGIS](https://img.shields.io/badge/database-PostgreSQL%2FPostGIS-blue.svg)

**InfraScan** is an enterprise-grade structural health monitoring platform designed to automate civil infrastructure inspection, defect localization, fine crack segmentation, surface characterization, and damage severity evaluation.

By combining real-time object detection ($YOLOv8$/$YOLO11$), dense semantic segmentation ($U\text{-}Net$), explainable AI ($Grad\text{-}CAM$), and spatial vector indexing ($PostGIS$), InfraScan transforms manual visual inspection workflows into a precise, data-driven computational pipeline.

---

## 🚀 Key Features

- **Multi-Role Authentication & Security Gateway:** OAuth2 + JWT token-based authentication supporting 5 operational roles (*Administrator, Infrastructure Manager, Civil Engineer, Field Inspector, Maintenance Engineer*) with immutable audit logging.
- **Image Quality Control Workbench:** Real-time variance of Laplacian blur detection ($\text{Var}(\Delta I) \ge 100.0$), dynamic range brightness checks, CLAHE contrast enhancement, and image deduplication.
- **AI Analytics & Visual Viewport:** Interactive multi-tab viewport rendering raw RGB imagery, macro bounding box defect detections, micro-crack segmentation masks, and Grad-CAM feature attribution overlays.
- **Metric Defect Quantification:** Automated estimation of physical crack length ($mm$), maximum width ($mm$), surface crack density ($\%$), and engineering severity ratings.
- **Geospatial Damage Mapping:** Interactive Leaflet GIS map displaying color-coded defect severity pins (Green, Yellow, Orange, Red) with temporal progression tracking.

---

## 🏗️ Core System Architecture

```text
[ Field Ingestion (UAV / Mobile / Drone) ]
                   │
                   ▼
       [ FastAPI Ingress Gateway ] ──(JWT Auth / RBAC)
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
[ Redis Queue ]           [ PostgreSQL / PostGIS ]
    │                             │
    ▼                             ▼
[ PyTorch GPU Worker ]    [ Structural Assets DB ]
 (YOLOv8 / U-Net / XAI)           │
    │                             │
    └──────────────┬──────────────┘
                   ▼
     [ React Dashboard Viewports ]
```

---

## 📊 Benchmark Datasets Supported

- **SDNET2018:** Concrete structural surface sub-image classification ($256\times256\text{ px}$).
- **RDD2022:** Multi-national road damage object detection (`D00` Longitudinal, `D10` Transverse, `D20` Alligator, `D40` Pothole).
- **CRACK500 & CrackForest (CFD):** High-resolution pixel-level crack segmentation masks.

---

## 🛠️ Quick Start Guide

### Prerequisites
- Python 3.10+
- Node.js v18+ LTS
- PostgreSQL 15+ with PostGIS extension enabled
- Docker Engine & Docker Compose (Optional)

### Containerized Setup (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/your-username/infrascan.git
cd infrascan

# 2. Launch all microservices with Docker Compose
docker-compose up --build -d
```

### Manual Development Setup

#### 1. Backend Setup (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Frontend Setup (React.js)
```bash
cd frontend
npm install
npm start
```

---

## 📋 System Environment Variables

Create a `.env` file in the root directory:

```env
PROJECT_NAME="InfraScan API"
SECRET_KEY="your-super-secret-jwt-key"
DATABASE_URL="postgresql://infrascan_user:password@localhost:5432/infrascan_db"
REDIS_URL="redis://localhost:6379/0"
STORAGE_PROVIDER="S3"
S3_BUCKET_NAME="infrascan-inspection-assets"
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
