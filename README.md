# Voice-to-Action AI Pipeline

An end-to-end AI system that converts natural language logistics queries into structured system actions.

##  Features
- Intent classification (booking, rates, payment, tracking, complaints)
- Entity extraction (locations, weight, packages, payment mode)
- Rule-based action decision engine
- FastAPI backend
- Streamlit frontend UI

##  Tech Stack
Python · scikit-learn · spaCy · FastAPI · Streamlit

##  Run Locally

**Start backend**
```bash
uvicorn main:app --reload
streamlit run streamlit_app.py
