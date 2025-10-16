# ⚡ Quick Start Guide

## 🚀 Start in 3 Steps

### 1️⃣ Start Backend (Terminal 1)
```bash
cd d:/biometric-text-analysis/backend
source ../venv/Scripts/activate  # Windows: venv\Scripts\activate
uvicorn app:app --reload --port 8000
```
✅ Backend running at: **http://localhost:8000**

### 2️⃣ Start Frontend (Terminal 2)
```bash
cd d:/biometric-text-analysis/frontend/widget
npm run dev
```
✅ Frontend running at: **http://localhost:3000**

### 3️⃣ Test It
Open browser: **http://localhost:3000**
Or test API: `curl http://localhost:8000/health`

---

## 🧪 Run Tests

### Quick Smoke Test (3 tests, ~4 seconds)
```bash
cd d:/biometric-text-analysis/backend
python -m pytest tests/smoke_test.py -v
```

### Full Test Suite (103 tests, ~15 seconds)
```bash
python -m pytest tests/ -v
```

---

## 📊 Current Status

✅ **103/103 tests passing** (100%)
- Encoder: 17 ✅
- Features: 12 ✅
- Normalizer: 42 ✅
- LLM Detector: 29 ✅
- Smoke Tests: 3 ✅

---

## 🔗 Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/verify` | POST | Authenticate user |
| `/enroll/start` | POST | Begin enrollment |
| `/enroll/submit` | POST | Submit sample |
| `/challenge/prepare` | POST | Get challenge |
| `/challenge/submit` | POST | Submit response |

---

## 📖 Documentation

- **README.md** - Full documentation (18KB)
- **PROJECT_STATUS.md** - Current status (6.4KB)
- **PROJECT_COMPLETION_SUMMARY.md** - Implementation details (20KB)
- **TASK.md** - Original requirements (18KB)

---

## 🛠️ Tech Stack

- **Backend:** FastAPI 0.119.0 + Python 3.12.4
- **Frontend:** React 18.2.0 + Vite 5.4.20
- **ML Model:** XLM-RoBERTa-base
- **Testing:** pytest 7.4.3

---

## ⚠️ Troubleshooting

### Backend won't start?
```bash
# Activate virtual environment first
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend errors?
```bash
cd frontend/widget
npm install
npm run dev
```

### Tests failing?
```bash
cd backend
python -m pytest tests/ -v
```

---

## 🎯 What's Working

✅ All 6 API endpoints functional
✅ XLM-RoBERTa text embeddings
✅ Keystroke biometric analysis
✅ LLM detection (7 signals)
✅ React widget with paste prevention
✅ 103 tests passing (100%)

---

## 🎊 You're Ready!

Everything is configured and tested. Just start the servers and go!

**Happy Coding! 🚀**
