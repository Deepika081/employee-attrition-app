#  Employee Attrition Prediction System

An end-to-end machine learning system that predicts employee attrition and provides interpretable explanations using SHAP.

Built to demonstrate practical ML deployment — from experimentation to API integration, containerization, and interactive frontend.

---

##  Description

This project predicts whether an employee is likely to **Stay** or **Leave**, using a trained ML pipeline.

Beyond prediction, it provides:

- Leave probability (as a percentage)
- Risk level classification (Low / Medium / High)
- Top 3 feature contributions using SHAP (with direction of impact)

The goal of this project is to move beyond notebooks and demonstrate:

- Model serving with FastAPI
- Explainable AI using SHAP
- Interactive frontend integration with Gradio
- Containerization using Docker
- Multi-service orchestration with Docker Compose

---

##  Demo

Example Output:

```
🔹 Prediction: Stay
🔹 Leave Probability: 37%
🔹 Risk Level: Low

Top 3 Factors Influencing Prediction:
  ↑ average_montly_hours (0.347)
  ↓ last_evaluation (-0.147)
  ↓ satisfaction_level (-0.074)
```

![alt text](<gradio_demo.png>)

---

##  Docker Deployment (Recommended)

Run the entire system using Docker and Docker Compose.

### Prerequisites

- Docker Desktop installed
- Docker Compose enabled

---

###  Run with Docker Compose

From the project root:

```bash
docker compose up --build
```

This will:

- Build backend (FastAPI) container
- Build frontend (Gradio) container
- Create a shared internal Docker network
- Start both services

---

###  Access the Application

Frontend (Gradio UI):

```
http://localhost:7860
```

Backend API Docs (FastAPI Swagger):

```
http://localhost:8000/docs
```

---

###  Stop Containers

```bash
docker compose down
```

---

##  Local Installation (Without Docker)

Clone the repository:

```bash
git clone https://github.com/Deepika081/employee-attrition-app.git
cd employee-attrition-app
```

Create a virtual environment:

```bash
# Linux / Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠 Local Usage

###  Run Backend (FastAPI)

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs available at:

```
http://127.0.0.1:8000/docs
```

---

###  Run Frontend (Gradio)

```bash
python frontend/app.py
```

Open in browser:

```
http://127.0.0.1:7860
```

Enter employee data in the UI to receive predictions and interpretability insights.

---

##  Features

- Binary classification: Stay vs Leave
- Leave probability as percentage
- Risk level categorization
- SHAP-based top 3 feature explanations
- Preprocessing + model pipeline integration
- FastAPI backend for model serving
- Gradio UI for interactive testing
- Dockerized multi-container architecture
- Clean modular project structure (backend / frontend / experiments)

---

##  Tech Stack

- Python
- Scikit-learn
- SHAP
- Pandas
- NumPy
- FastAPI
- Gradio
- Docker
- Joblib

---

##  Architecture Overview

1. Model trained and validated in Jupyter (EDA + preprocessing + pipeline)
2. Saved as a serialized pipeline (`joblib`)
3. Loaded at FastAPI startup
4. SHAP TreeExplainer initialized for interpretability
5. Gradio frontend sends user input to backend
6. Backend returns prediction + probability + top contributing features
7. Docker Compose orchestrates frontend and backend containers

---

##  Project Structure

```
employee-attrition-app/
│
├── backend/
│   ├── main.py
│   ├── pipeline.joblib
│   ├── Dockerfile
│
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│
├── experiments/
│   ├── experimental_code.ipynb
│
├── docker-compose.yml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📄 License

This project is licensed under the MIT License.