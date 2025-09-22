from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field
import joblib
import logging
import numpy as np
from typing import Optional, List, Dict, Any
import traceback
from datetime import datetime, timedelta
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insurance Risk Assessment API",
    description="Comprehensive ML-powered insurance risk analysis system",
    version="2.0.0"
)

# Risk categories
class RiskCategory(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium" 
    LOW = "Low"

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

# Load models with error handling
try:
    # Core risk scoring model
    risk_model = joblib.load("risk_model.pkl")
    training_columns = joblib.load("training_columns.pkl")
    
    # Additional models for comprehensive analysis
    claim_probability_model = joblib.load("claim_probability_model.pkl")  # Returns probability 0-1
    expected_loss_model = joblib.load("expected_loss_model.pkl")  # Predicts loss amount
    
    # Risk thresholds and parameters
    risk_thresholds = {
        "low_threshold": 0.3,    # Below this = Low risk
        "high_threshold": 0.7,   # Above this = High risk
        "company_risk_capacity": 50000000,  # $50M company risk capacity
        "reinsurance_threshold": 0.8  # If portfolio risk > 80% of capacity, suggest reinsurance
    }
    
    logger.info("All models loaded successfully")
    
except FileNotFoundError as e:
    logger.error(f"Model files not found: {e}")
    risk_model = None
    claim_probability_model = None
    expected_loss_model = None
    training_columns = None
    risk_thresholds = None
except Exception as e:
    logger.error(f"Error loading models: {e}")
    risk_model = None
    claim_probability_model = None
    expected_loss_model = None
    training_columns = None
    risk_thresholds = None

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_models_loaded():
    """Check if all models are loaded properly"""
    if any(model is None for model in [risk_model, claim_probability_model, expected_loss_model, training_columns]):
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. ML models not loaded properly."
        )

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardized data preprocessing"""
    try:
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=training_columns, fill_value=0)
        return df_encoded
    except Exception as e:
        logger.error(f"Data preprocessing error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Error processing data: {str(e)}"
        )

def get_risk_category(risk_score: float) -> RiskCategory:
    """Convert numerical risk score to category"""
    if risk_score >= risk_thresholds["high_threshold"]:
        return RiskCategory.HIGH
    elif risk_score >= risk_thresholds["low_threshold"]:
        return RiskCategory.MEDIUM
    else:
        return RiskCategory.LOW

def calculate_claim_probability(df_encoded: pd.DataFrame, years: int = 1) -> float:
    """Calculate probability of claim within X years"""
    try:
        base_probability = claim_probability_model.predict_proba(df_encoded)[0][1]  # Probability of claim=1
        # Adjust for time horizon (simple exponential model)
        adjusted_probability = 1 - (1 - base_probability) ** years
        return float(adjusted_probability)
    except Exception as e:
        logger.error(f"Claim probability calculation error: {e}")
        return 0.0

def calculate_expected_loss(df_encoded: pd.DataFrame, sum_insured: float, claim_probability: float) -> Dict[str, float]:
    """Calculate expected loss forecasting"""
    try:
        # Get severity (expected loss amount if claim occurs)
        expected_severity = expected_loss_model.predict(df_encoded)[0]
        
        # Cap severity at sum insured
        capped_severity = min(float(expected_severity), sum_insured)
        
        # Expected loss = Probability × Severity
        expected_loss = claim_probability * capped_severity
        
        return {
            "expected_severity": round(capped_severity, 2),
            "expected_loss": round(expected_loss, 2),
            "loss_ratio": round((expected_loss / sum_insured) * 100, 2)  # As percentage
        }
    except Exception as e:
        logger.error(f"Expected loss calculation error: {e}")
        return {
            "expected_severity": 0.0,
            "expected_loss": 0.0,
            "loss_ratio": 0.0
        }

def calculate_capacity_allocation(portfolio_exposure: float, new_policy_exposure: float) -> Dict[str, Any]:
    """Calculate capacity allocation recommendations"""
    try:
        total_exposure = portfolio_exposure + new_policy_exposure
        capacity_utilization = total_exposure / risk_thresholds["company_risk_capacity"]
        
        recommendations = {
            "current_capacity_utilization": round(capacity_utilization * 100, 2),
            "total_exposure": round(total_exposure, 2),
            "available_capacity": round(risk_thresholds["company_risk_capacity"] - total_exposure, 2),
            "needs_reinsurance": capacity_utilization > risk_thresholds["reinsurance_threshold"],
            "recommended_retention": 0,
            "recommended_reinsurance": 0,
            "action": "accept"
        }
        
        if recommendations["needs_reinsurance"]:
            # Suggest keeping exposure at 80% of capacity
            target_exposure = risk_thresholds["company_risk_capacity"] * 0.8
            recommended_retention = target_exposure - portfolio_exposure
            recommended_reinsurance = new_policy_exposure - recommended_retention
            
            recommendations.update({
                "recommended_retention": round(max(0, recommended_retention), 2),
                "recommended_reinsurance": round(max(0, recommended_reinsurance), 2),
                "action": "reinsure" if recommended_reinsurance > 0 else "accept"
            })
            
            if recommended_retention <= 0:
                recommendations["action"] = "decline"
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Capacity allocation calculation error: {e}")
        return {
            "current_capacity_utilization": 0,
            "total_exposure": 0,
            "available_capacity": 0,
            "needs_reinsurance": False,
            "recommended_retention": 0,
            "recommended_reinsurance": 0,
            "action": "error"
        }

# Enhanced Policy model
class Policy(BaseModel):
    location: str
    yearBuilt: int = Field(ge=1800, le=2025, description="Year the property was built")
    Constructon: str = Field(description="Construction type")
    SumInsured: float = Field(gt=0, description="Sum insured amount")
    
    # Optional fields for enhanced analysis
    previous_claims: Optional[int] = Field(default=0, ge=0, description="Number of previous claims")
    property_age: Optional[int] = None
    risk_features: Optional[Dict[str, Any]] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Calculate property age
        if self.property_age is None:
            self.property_age = datetime.now().year - self.yearBuilt

class PortfolioAnalysis(BaseModel):
    current_portfolio_exposure: float = Field(gt=0, description="Current portfolio total exposure")
    analysis_years: Optional[int] = Field(default=1, ge=1, le=10, description="Years for probability analysis")

class ComprehensiveAnalysisRequest(BaseModel):
    policy: Policy
    portfolio: Optional[PortfolioAnalysis] = None

@app.post("/comprehensive-analysis/")
async def comprehensive_analysis(request: ComprehensiveAnalysisRequest):
    """
    Complete risk analysis including:
    1. Risk scoring (High/Medium/Low)
    2. Claim probability estimation
    3. Expected loss forecasting  
    4. Capacity allocation recommendations
    """
    try:
        validate_models_loaded()
        
        policy = request.policy
        portfolio = request.portfolio or PortfolioAnalysis(current_portfolio_exposure=0)
        
        # Prepare data for ML models
        df = pd.DataFrame([policy.model_dump()])
        df_encoded = preprocess_data(df)
        
        # 1. Risk Scoring
        risk_score = float(risk_model.predict(df_encoded)[0])
        risk_category = get_risk_category(risk_score)
        
        # 2. Claim Probability Estimation
        claim_probability = calculate_claim_probability(df_encoded, portfolio.analysis_years)
        
        # 3. Expected Loss Forecasting
        loss_analysis = calculate_expected_loss(df_encoded, policy.SumInsured, claim_probability)
        
        # 4. Capacity Allocation
        capacity_analysis = calculate_capacity_allocation(
            portfolio.current_portfolio_exposure,
            policy.SumInsured
        )
        
        # Compile comprehensive response
        response = {
            "policy_id": f"POL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "analysis_timestamp": datetime.now().isoformat(),
            "policy_details": policy.model_dump(),
            
            # Risk Scoring
            "risk_assessment": {
                "risk_score": round(risk_score, 3),
                "risk_category": risk_category.value,
                "confidence": round(min(abs(risk_score - 0.5) * 2, 1.0), 3)  # Higher confidence for scores further from 0.5
            },
            
            # Claim Probability
            "claim_analysis": {
                "analysis_period_years": portfolio.analysis_years,
                "claim_probability": round(claim_probability * 100, 2),  # As percentage
                "annual_claim_rate": round(claim_probability / portfolio.analysis_years * 100, 2)
            },
            
            # Expected Loss
            "loss_forecasting": loss_analysis,
            
            # Capacity Allocation
            "capacity_management": capacity_analysis,
            
            # Overall Recommendation
            "recommendation": {
                "action": capacity_analysis["action"],
                "risk_tier": risk_category.value,
                "pricing_multiplier": 1.0 + (risk_score * 0.5),  # Simple pricing adjustment
                "monitoring_required": risk_category in [RiskCategory.MEDIUM, RiskCategory.HIGH],
                "notes": []
            }
        }
        
        # Add contextual notes
        notes = response["recommendation"]["notes"]
        if risk_category == RiskCategory.HIGH:
            notes.append("High risk policy requires additional underwriting review")
        if claim_probability > 0.5:
            notes.append(f"High claim probability ({claim_probability*100:.1f}%) within {portfolio.analysis_years} years")
        if capacity_analysis["needs_reinsurance"]:
            notes.append("Portfolio capacity limits require reinsurance consideration")
        if loss_analysis["loss_ratio"] > 80:
            notes.append("High expected loss ratio - consider premium adjustment")
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error performing comprehensive risk analysis"
        )

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    """Enhanced CSV processing with comprehensive analysis"""
    try:
        validate_models_loaded()
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Read and validate CSV
        try:
            df = pd.read_csv(file.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
            
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Process each row
        results = []
        total_portfolio_exposure = 0
        
        for index, row in df.iterrows():
            try:
                # Create policy from row
                policy_data = row.to_dict()
                
                # Preprocess data
                row_df = pd.DataFrame([policy_data])
                row_encoded = preprocess_data(row_df)
                
                # Get sum insured (adjust column name as needed)
                sum_insured = policy_data.get('SumInsured', 0)
                total_portfolio_exposure += sum_insured
                
                # Perform analysis
                risk_score = float(risk_model.predict(row_encoded)[0])
                risk_category = get_risk_category(risk_score)
                claim_probability = calculate_claim_probability(row_encoded, years=1)
                loss_analysis = calculate_expected_loss(row_encoded, sum_insured, claim_probability)
                
                # Add results to original data
                row_result = policy_data.copy()
                row_result.update({
                    'risk_score': round(risk_score, 3),
                    'risk_category': risk_category.value,
                    'claim_probability_1yr': round(claim_probability * 100, 2),
                    'expected_loss': loss_analysis['expected_loss'],
                    'loss_ratio': loss_analysis['loss_ratio']
                })
                
                results.append(row_result)
                
            except Exception as e:
                logger.error(f"Error processing row {index}: {e}")
                # Add error row
                error_result = row.to_dict()
                error_result.update({
                    'risk_score': 'ERROR',
                    'risk_category': 'ERROR',
                    'claim_probability_1yr': 'ERROR',
                    'expected_loss': 'ERROR',
                    'loss_ratio': 'ERROR',
                    'error': str(e)
                })
                results.append(error_result)
        
        # Portfolio level analysis
        portfolio_stats = {
            "total_policies": len(results),
            "total_exposure": round(total_portfolio_exposure, 2),
            "avg_risk_score": round(np.mean([r['risk_score'] for r in results if isinstance(r['risk_score'], (int, float))]), 3),
            "risk_distribution": {
                "high": len([r for r in results if r.get('risk_category') == 'High']),
                "medium": len([r for r in results if r.get('risk_category') == 'Medium']),
                "low": len([r for r in results if r.get('risk_category') == 'Low'])
            },
            "total_expected_loss": round(sum([r['expected_loss'] for r in results if isinstance(r['expected_loss'], (int, float))]), 2)
        }
        
        return {
            "status": "success",
            "portfolio_summary": portfolio_stats,
            "detailed_results": results,
            "columns": list(results[0].keys()) if results else [],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV upload error: {e}")
        raise HTTPException(status_code=500, detail="Error processing CSV file")

@app.get("/health")
async def health_check():
    """Enhanced health check with model status"""
    try:
        model_status = {
            "risk_model": "loaded" if risk_model is not None else "not loaded",
            "claim_probability_model": "loaded" if claim_probability_model is not None else "not loaded",
            "expected_loss_model": "loaded" if expected_loss_model is not None else "not loaded",
            "training_columns": "loaded" if training_columns is not None else "not loaded"
        }
        
        all_loaded = all(status == "loaded" for status in model_status.values())
        
        return {
            "status": "healthy" if all_loaded else "partially available",
            "models": model_status,
            "capabilities": {
                "risk_scoring": model_status["risk_model"] == "loaded",
                "claim_probability": model_status["claim_probability_model"] == "loaded",
                "expected_loss": model_status["expected_loss_model"] == "loaded",
                "comprehensive_analysis": all_loaded
            },
            "risk_thresholds": risk_thresholds
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )