#for the model training we will approach it with three steps

#STEP 1 - GENERATE SYNTHETIC INSURANCE DATASET SE WE DON'T HAVE ANY

#This synthetic data lets you test whether machine learning models can automatically
#discover the same risk patterns that human underwriters use!

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

samples = 500
locations = ["Nairobi", "Mombasa", "Kisumu", "Eldoret", "Nakuru", "Thika"]
constructions = ["Concrete", "Timber", "Masonry", "Mud"]

data = []
for i in range(samples):
    loc = np.random.choice(locations)
    year_built = np.random.randint(1980, 2023)
    construction = np.random.choice(constructions)
    sum_insured = np.random.randint(100_000, 2_000_000)

    #RULES only here (to simulate ground truth risk)
    base_risk = 0.3 

    if loc == "Nairobi":
        base_risk -= 0.15  # Nairobi is safer (-15%)
    elif loc in ["Eldoret", "Nakuru", "Thika"]:
        base_risk -= 0.05  # Other inland cities slightly safer (-5%)
    elif loc in ["Mombasa", "Kisumu"]:
        base_risk += 0.20

    if construction == "Concrete":
        base_risk -= 0.10  # Concrete is strongest (-10%)
    elif construction == "Masonry":
        base_risk -= 0.05  # Masonry is decent (-5%)
    elif construction == "Timber":
        base_risk += 0.15  # Timber is fire-prone (+15%)
    elif construction == "Mud":
        base_risk += 0.25
    
    if sum_insured <= 300_000:      # Low value
        base_risk -= 0.05
    elif sum_insured <= 1_000_000:  # Medium value
        base_risk += 0.00  # No change
    else:                           # High value
        base_risk += 0.10

    # add noise of 5% to make it realistic
    risk_score = min(0.9, max(0.05, base_risk + np.random.normal(0, 0.03)))

    data.append([loc, year_built, construction, sum_insured, risk_score])

df = pd.DataFrame(data, columns=["Location", "YearBuilt", "Construction", "SumInsured", "RiskScore"])
df.head()

#STEP 2 - TRAINING THE REGRESSION MODEL

# coverting One-hot encode categorical features for the model to understand
df_encoded = pd.get_dummies(df, columns=["Location", "Construction"])

X = df_encoded.drop("RiskScore", axis=1) #all the input variables
y = df_encoded["RiskScore"] #what we are trying to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split dataset 20% for testing

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained successfully")
print("Model R² on test:", model.score(X_test, y_test))

#STEP 3 - SAVE MODEL AND TRAINING COLUMNS
joblib.dump(model, "risk_model.pkl")
joblib.dump(list(X.columns), "training_columns.pkl")

#----------------------------
# CAVEATS
#----------------------------

training_columns = X.columns


print("\n" + "=" * 50)
print("Testing with realistic scenarios:")
print("=" * 50)

test_cases = [
    {
        "Location": "Nairobi",
        "YearBuilt": 2020,
        "Construction": "Concrete", 
        "SumInsured": 500_000,
        "Expected": "LOW risk (safe city + strong material + new building)"
    },
    {
        "Location": "Nairobi", 
        "YearBuilt": 2020,
        "Construction": "Timber",
        "SumInsured": 750_000,
        "Expected": "MEDIUM risk (safe city + weak material + new building)"
    },
    {
        "Location": "Mombasa",
        "YearBuilt": 1990, 
        "Construction": "Timber",
        "SumInsured": 1_500_000,
        "Expected": "HIGH risk (risky city + weak material + old building + high value)"
    },
    {
        "Location": "Mombasa",
        "YearBuilt": 2022,
        "Construction": "Concrete", 
        "SumInsured": 300_000,
        "Expected": "MEDIUM risk (risky city offset by strong material + new building + low value)"
    }
]

loaded_model = joblib.load("risk_model.pkl")

for i, case in enumerate(test_cases, 1):
    # Create policy dataframe
    new_policy_raw = pd.DataFrame([{
        "Location": case["Location"],
        "YearBuilt": case["YearBuilt"],
        "Construction": case["Construction"],
        "SumInsured": case["SumInsured"]
    }])
    
    # Encode and predict
    new_policy_encoded = pd.get_dummies(new_policy_raw)
    new_policy_encoded = new_policy_encoded.reindex(columns=training_columns, fill_value=0)
    predicted_risk = loaded_model.predict(new_policy_encoded)[0]
    
    # Determine risk level
    if predicted_risk < 0.25:
        risk_level = "LOW"
    elif predicted_risk < 0.45:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    print(f"\nTest Case {i}:")
    print(f"Policy: {case['Location']}, {case['YearBuilt']}, {case['Construction']}, KSh {case['SumInsured']:,}")
    print(f"Predicted Risk: {predicted_risk:.3f} ({risk_level})")
    print(f"Expected: {case['Expected']}")

# Show feature importance
print("\n" + "=" * 50)
print("Feature Importance (what the model learned):")
print("=" * 50)
feature_importance = pd.DataFrame({
    'feature': training_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:<30} {row['importance']:.3f}")