from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
import random

app = FastAPI()

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

    #adding fake AI risk scores for now
    dataFrame["AI_Risk_Sore"] = [
        random.choice(["Low", "Medium", "High"]) for _ in range(len(dataFrame))
    ]

    #return the first few rows just to confirm it works
    return {
        "columns": dataFrame.columns.tolist(),
        "rows": dataFrame.to_dict(orient="records")
    }



