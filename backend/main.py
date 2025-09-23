from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
import joblib
import logging
import numpy as np

portfolio_df = None

app = FastAPI(
    title="Insurance Risk Assessment API",
    description="Comprehensive ML-powered insurance risk analysis system",
    version="1.0.0"
)

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

def categorize_risk(score: float) -> str:
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Medium"
    else:
        return "High"

def compute_metrics(score: float, sum_insured: float, construction: str = None) -> dict:
    """Convert raw model score into risk insights."""
    claim_probability = float(
        min(0.95, max(0.05, score + np.random.normal(0, 0.05)))
    )

    if construction == "Concrete":
        severity = 0.2
    elif construction == "Timber":
        severity = 0.5
    elif construction == "mud":
        severity = 0.6
    else:
        severity = 0.35


    expected_loss = round(claim_probability * severity * sum_insured, 2)
    capacity_flag = "Above Threshold" if score > 0.6 else "Within Capacity"

    return {
        "predicted_risk": round(float(score), 3),
        "category": categorize_risk(score),
        "claim_probability": round(claim_probability, 3),
        "expected_loss": expected_loss,
        "capacity_flag": capacity_flag,
    }

def portfolio_summary(df: pd.DataFrame) -> dict:
    """Aggregate portfolio-level metrics."""

    # Category distribution (pie chart)
    category_distribution = df["category"].value_counts(normalize=True).round(3).to_dict()
    pie_data = [{"label": cat, "value": val} for cat, val in category_distribution.items()]

    # Capacity flags (bar chart)
    capacity_flags = df["capacity_flag"].value_counts().to_dict()
    bar_data = [{"label": flag, "value": count} for flag, count in capacity_flags.items()]

    # Claim probability trend (histogram style)
    claim_prob_bins = pd.cut(df["claim_probability"], bins=[0, 0.25, 0.5, 0.75, 1.0])
    claim_prob_dist = claim_prob_bins.value_counts().sort_index()
    line_data = [
        {"range": str(interval), "count": int(count)}
        for interval, count in claim_prob_dist.items()
    ]

    # --- loss distributin (Histogram of expected losses) ----
    loss_bins = pd.cut(df["expected_loss"], bins=5)
    loss_dist = loss_bins.value_counts().sort_index()
    loss_hist = [
        {"range": str(interval), "count": int(count)}
        for interval, count in loss_dist.items()
    ]

    return {
        "totalPolicies": int(len(df)),
        "avgRiskScore": round(df["predicted_risk"].mean(), 3),
        "avgClaimProbability": round(df["claim_probability"].mean(), 3),
        "totalExpectedLoss": round(df["expected_loss"].sum(), 2),
        "averageExpectedLoss": round(df["expected_loss"].mean(), 2),

        # Raw aggregates
        "categoryDistribution": category_distribution,
        "capacityFlags": capacity_flags,

        # Chart-ready data
        "charts": {
            "category_pie": pie_data,
            "capacity_bar": bar_data,
            "claim_prob_line": line_data,
            "lossHistogram": loss_hist,
        }
    }

class AdjustmentRequest(BaseModel):
    adjustments: dict #we are expecting a dictionary adjustment

    #columns - keys
    #values - mapping conditions

#ENDPOINTS
#-------------------------
@app.post("/scenario-test/")
def scenario_test(request: AdjustmentRequest):

    global portfolio_df
    if portfolio_df is None:
        return {"error": "No portfolio uploaded yet."}

    dataFrame = portfolio_df.copy()

    adjusted_df = dataFrame.copy()
    for column, rules in request.adjustments.items(): #loops through columns and its rules
        for key, change in rules.items(): #loops through the keys and unit change 
            if column not in dataFrame:
                continue
            
        if column == "YearBuilt" and key.startswith("<"):
            year = int(key[1:])
            mask = dataFrame[column] < year
        elif column == "YearBuilt" and key.startswith(">"):
            year = int(key[1:])
            mask = dataFrame[column] > year
        elif column == "YearBuilt":
            year = int(key)
            mask = dataFrame[column] == year
        else:
            mask = dataFrame[column] == key
            #Apply increase in risk probability
            adjusted_df.loc[mask, "claim_probability"] *= (1 + change)
    #clip between 0-1
    adjusted_df["claim_probability"] = adjusted_df["claim_probability"].clip(0, 1)

    # Recompute expected loss
    adjusted_df["expected_loss"] = (
        adjusted_df["claim_probability"] * adjusted_df["SumInsured"]
    )

    # Use your existing portfolio_summary function
    summary = portfolio_summary(adjusted_df)

    return {
        "baseline": portfolio_summary(dataFrame), #original
        "scenario": summary #with adjustments
    }


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):

    dataFrame = pd.read_csv(file.file) #An instance for storing data
    global portfolio_df
    

    dataFrameEncoded = pd.get_dummies(dataFrame)
    dataFrameEncoded = dataFrameEncoded.reindex(columns=training_columns, fill_value=0)


    scores = model.predict(dataFrameEncoded)

    results = []
    for i, score in enumerate(scores):
        metrics = compute_metrics(score, dataFrame.iloc[i]["SumInsured"], dataFrame.iloc[i]["Construction"])
        results.append(metrics)

    for key in results[0].keys():
        dataFrame[key] = [r[key] for r in results]
    
    summary = portfolio_summary(dataFrame)
    portfolio_df = dataFrame.copy() #A copy to use in what-if situations
    
    return {
        "columns": dataFrame.columns.tolist(),
        "rows": dataFrame.to_dict(orient="records"),
        "summary": summary
    }


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

    score = model.predict(df_encoded)[0]
    return compute_metrics(score, df["SumInsured"].iloc[0],  df["Construction"].iloc[0])



