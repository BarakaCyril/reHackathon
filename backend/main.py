from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
import random
import joblib

app = FastAPI()
model = joblib.load("risk_model.pkl")

training_columns = joblib.load("training_columns.pkl")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production: restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):

    dataFrame = pd.read_csv(file.file) #An instance for storing data

    dataFrameEncoded = pd.get_dummies(dataFrame)
    dataFrameEncoded = dataFrameEncoded.reindex(columns=training_columns, fill_value=0)
    #adding fake AI risk scores for now


    dataFrame["AI_Risk_Sore"] = model.predict(dataFrameEncoded)
    #return the first few rows just to confirm it works
    return {
        "columns": dataFrame.columns.tolist(),
        "rows": dataFrame.to_dict(orient="records")
    }


#--------------------
#PREDICTING PER ROW POLICY

class Policy(BaseModel):
    location: str
    yearBuilt: int
    Construction: str
    SumInsured: float

@app.post("/predict-policy/")
async def predict_policy(policy: Policy):
    df = pd.DataFrame([policy.model_dump()])

    # One-hot encode and align with training columns
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=training_columns, fill_value=0)

    predicted_risk = model.predict(df_encoded)[0]

    return {
        "predicted_risk": round(float(predicted_risk), 3)
    }


