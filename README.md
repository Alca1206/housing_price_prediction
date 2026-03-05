# 🏠 King County House Price Prediction AI

This project is a full-stack machine learning application designed to estimate real estate prices in King County, Washington. It leverages a custom Deep Learning model built with **PyTorch**, serves predictions through a lightning-fast REST API using **FastAPI**, and provides a user-friendly, interactive web dashboard built with **Streamlit**.



## 🌟 Project Architecture
* **Model:** Deep Learning Neural Network built with PyTorch (Feedforward Network).
* **Backend:** REST API built with FastAPI.
* **Frontend:** Interactive web dashboard built with Streamlit.
* **Deployment (MLOps):** Fully containerized and orchestrated using Docker & Docker Compose for guaranteed reproducibility.

## 📂 Repository Structure
```text
king-county-house-prices/
│
├── .dockerignore                # Ignored files for Docker build
├── docker-compose.yml           # Orchestrates the multi-container application
├── Dockerfile                   # Blueprint for the Python environment
├── .gitignore                   # Ignored files (__pycache__, etc.)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── notebooks/                   
│   └── housing_price_prediction.ipynb  # Data exploration and model training
│
├── model/                       
│   └── full_model_package.pkl   # Bundled PyTorch weights, Scaler, and Columns
│
└── src/                         
    ├── main.py                  # FastAPI Backend Server
    └── app.py                   # Streamlit Frontend UI
```
## 🐳 How to Run with Docker (Recommended)
This project is fully containerized using Docker Compose, ensuring a 100% reproducible environment across any machine.

### 1. Build and Start the Containers
Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running. Open your terminal in the root directory of the project and run:

```bash
docker-compose up --build
```

### 2. Access the Application
Docker will automatically build the images, install the required dependencies (using a lightweight CPU-only PyTorch build), and spin up both services simultaneously:

Streamlit Frontend: Open your browser and go to http://localhost:8501

FastAPI Interactive Docs: Open your browser and go to http://localhost:8000/docs

### 3. Stop the Containers
When you are finished, simply press Ctrl + C in your terminal, or run:
```
bash
docker-compose down
```

## 📋 Requirements
- Python 3.8+
- pip or conda package manager

## 🚀 Installation & Setup

### 1. Clone & Install Dependencies
```bash
git clone <repository-url>
cd housing_price_prediction
pip install -r requirements.txt
```
## ▶️ Running the Project

### Option A: Full Web Application (Recommended)
Start both the FastAPI backend and Streamlit frontend:

**Terminal 1 - FastAPI Backend:**
```bash
cd src
uvicorn main:app --reload
```
API docs available at: `http://127.0.0.1:8000/docs`

**Terminal 2 - Streamlit Frontend:**
```bash
cd src
streamlit run app.py
```
Web UI opens automatically at: `http://localhost:8501`

### Option B: FastAPI Only (for API calls)
```bash
cd src
uvicorn main:app --reload
```
Use tools like Postman or curl to interact with the API at `http://127.0.0.1:8000`

## 🧠 Model Details

The model and preprocessing pipeline are bundled in `model/full_model_package.pkl` (joblib format):

| Component | Description |
|-----------|-------------|
| `model_state` | Trained PyTorch neural network weights |
| `scaler` | Fitted StandardScaler for feature normalization |
| `columns` | Expected feature names and order |

**Model Architecture:**
- Input Layer: 16 features (after encoding)
- Hidden Layer 1: 64 neurons + ReLU
- Hidden Layer 2: 32 neurons + ReLU
- Output Layer: 1 neuron (price prediction)

## 💻 Custom Inference (Python Script)

Load the model in a custom script or Jupyter Notebook without the web app:

```python
import joblib
import torch
import torch.nn as nn
import pandas as pd

# 1. Define the PyTorch model architecture
class HousePricePredictor(nn.Module):
    def __init__(self, input_dim):
        super(HousePricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# 2. Load the bundled model package
package = joblib.load("../model/full_model_package.pkl")
scaler = package['scaler']
model_columns = package['columns']
state_dict = package['model_state']

# 3. Initialize and load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HousePricePredictor(input_dim=len(model_columns))
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 4. Make a prediction
sample_house = pd.DataFrame([{
    "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1500, 
    "sqft_lot": 5000, "floors": 1.0, "zipcode": "98001",
    "waterfront": 0, "view": 0, "condition": 3, "grade": 7,
    "sqft_basement": 0, "yr_built": 1990, "yr_renovated": 0,
    "lat": 47.5, "long": -122.2, "sqft_living15": 1500, "sqft_lot15": 5000
}])

# Preprocess: encode categorical + align with training columns + scale
sample_encoded = pd.get_dummies(sample_house, columns=["zipcode"], drop_first=False)
sample_aligned = sample_encoded.reindex(columns=model_columns, fill_value=0)
sample_scaled = scaler.transform(sample_aligned)

# Run inference
input_tensor = torch.FloatTensor(sample_scaled).to(device)
with torch.no_grad():
    predicted_price = model(input_tensor)

print(f"Predicted House Price: ${predicted_price.item():,.2f}")
```

## 📊 Dataset
- **Source:** https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
- **Features:** 16 (bedrooms, bathrooms, sqft_living, zipcode, etc.)
- **Target:** Price (continuous)
- **Samples:** ~21,613 records

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'fastapi'` | Run `pip install -r requirements.txt` |
| Port 8000 already in use | Use `uvicorn main:app --port 8001 --reload` |
| Port 8501 already in use | Kill the process or use `streamlit run app.py --server.port 8502` |
| Model prediction mismatch | Ensure features are in the exact order from `model_columns` |
| CUDA/GPU errors | Set `device = "cpu"` or ensure CUDA toolkit is installed |

## 📚 Additional Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)



## 👤 Author
Dev Gupta
