from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from .models import Predict_winner
import matplotlib.pyplot as plt
import os

def train_model():
    matches = Predict_winner.objects.all()
    if not matches:
        return None
    
    data = []
    for match in matches:
        data.append([
            hash(match.team1),
            hash(match.team2),
            hash(match.venue),
            match.score1,
            match.wickets1,
            match.balls_left1,
            match.score2,
            match.wickets2,
            match.balls_left2,
            1 if match.winner == match.team1 else (0 if match.winner == "Draw" else -1)
        ])

    df = pd.DataFrame(data, columns=[
        "team1", "team2", "venue", "score1", "wickets1", "balls_left1",
        "score2", "wickets2", "balls_left2", "result"
    ])

    feature_columns = ["team1", "team2", "venue", "score1", "wickets1", "balls_left1", "score2", "wickets2", "balls_left2"]
    X = df[feature_columns].values  # Convert to NumPy array

    y = df["result"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Loss", "Draw", "Win"], yticklabels=["Loss", "Draw", "Win"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    cm_graph_path = os.path.join("pages", "static", "rf_confusion_matrix.png")
    plt.savefig(cm_graph_path)
    plt.close()

    return model


def predict_match(team1, team2, venue, present_score, wickets_left, balls_remaining):
    model = train_model()
    
    if model is None:
        return "Model training failed. No data available."

    # Encode categorical values (use same encoding as training)
    label_encoder = LabelEncoder()
    team1_encoded = label_encoder.fit_transform([team1])[0]
    team2_encoded = label_encoder.fit_transform([team2])[0]
    venue_encoded = label_encoder.fit_transform([venue])[0]

    # Since the model was trained with both innings data, we assume:
    score1 = present_score
    wickets1 = wickets_left
    balls_left1 = balls_remaining

    # Dummy values for second innings since we don't have it in prediction
    score2 = 0  
    wickets2 = 0  
    balls_left2 = 0  

    # Ensure input matches training feature count
    input_data = [[
        team1_encoded, team2_encoded, venue_encoded,
        score1, wickets1, balls_left1,
        score2, wickets2, balls_left2
    ]]

    # Predict the outcome
    prediction = model.predict(input_data)[0]

    # Interpret the result
    if prediction == 1:
        return f"{team1} is predicted to win."
    elif prediction == -1:
        return f"{team2} is predicted to win."
    else:
        return "Match is predicted to be a draw."

