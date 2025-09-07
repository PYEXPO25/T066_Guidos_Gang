import os
import django
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'T066_Guidos_Gang.settings')
django.setup()

# Import your model
from pages.models import PlayerPredict

# Load data from PlayerPredict model
qs = PlayerPredict.objects.all().values()
df = pd.DataFrame.from_records(qs)

# Check for empty dataset
if df.empty:
    print("⚠️ No data available in the PlayerPredict table.")
    exit()

# Encode categorical columns
label_encoders = {}
categorical_cols = ['venue', 'pitch_type', 'player_name']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define feature columns and target
feature_cols = ['strike_rate', 'runs', 'wickets', 'overs', 'venue', 'pitch_type', 'economy']
target_col = 'player_name'

X = df[feature_cols]
y = df[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define paths to save model and encoders
MODEL_PATH = os.path.join("pages", "prediction", "player_predict_model.pkl")
ENCODERS_PATH = os.path.join("pages", "prediction", "player_predict_encoders.pkl")

# Save the trained model and encoders using joblib
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoders, ENCODERS_PATH)

print("✅ Player prediction model trained and saved successfully!")
print(f"📁 Model saved at: {MODEL_PATH}")
print(f"📁 Encoders saved at: {ENCODERS_PATH}")
