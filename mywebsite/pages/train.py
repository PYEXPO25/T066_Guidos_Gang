import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pages.models import BattingFirstPredict  # Import Django model

# Fetch data from database
data = BattingFirstPredict.objects.all().values(
    "present_score", "balls_remaining", "wickets_left", "venue", "predict_target"
)

if not data:
    print("No data available in the database. Please insert records first.")
else:
    # Venue encoding
    venue_mapping = {
        "Wankhede Stadium": 1, "Chinnaswamy Stadium": 2, "Eden Gardens": 3,
        "Kotla": 4, "Chepauk": 5, "Arun Jaitley Stadium": 6,
        "MA Chidambaram Stadium": 7, "Rajiv Gandhi International Stadium": 8,
        "Narendra Modi Stadium": 9, "Sawai Mansingh Stadium": 10,
        "BRSABV Ekana Cricket Stadium": 11, "Punjab Cricket Association Stadium": 12
    }

    # Prepare data
    X, y = [], []
    for row in data:
        venue_encoded = venue_mapping.get(row["venue"], 0)
        X.append([row["present_score"], row["balls_remaining"], row["wickets_left"], venue_encoded])
        y.append(row["predict_target"])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_path = "C:/Users/raahu/Desktop/Django8/mywebsite/pages/prediction/ml_model.pkl"
    joblib.dump(model, model_path)

    print(f"Model saved successfully at: {model_path}")
