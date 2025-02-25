import os
import pickle
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt # type: ignore

import seaborn as sns # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
from .models import Predict_winner

MODEL_PATH = os.path.join("pages", "prediction", "rf_model.pkl")

def train_model(team1,team2):
    print("🚀 train_model() function started!")  # Debugging
    print(team2)

    matches = Predict_winner.objects.all()
    if not matches:
        return None
    
    data = []
    team_names = set()
    venue_names = set()

    for match in matches:
        team_names.update([match.team1, match.team2])
        venue_names.add(match.venue)
        data.append([
            match.team1, match.team2, match.venue,
            match.score1, match.wickets1, match.balls_left1,
            match.score2, match.wickets2, match.balls_left2,
            1 if match.winner == match.team1 else (0 if match.winner == "Draw" else -1)
        ])

    df = pd.DataFrame(data, columns=[
        "team1", "team2", "venue", "score1", "wickets1", "balls_left1",
        "score2", "wickets2", "balls_left2", "result"
    ])

    # Encoding
    team_encoder = LabelEncoder()
    venue_encoder = LabelEncoder()
    team_encoder.fit(list(team_names))
    venue_encoder.fit(list(venue_names))

    df["team1"] = team_encoder.transform(df["team1"])
    df["team2"] = team_encoder.transform(df["team2"])
    df["venue"] = venue_encoder.transform(df["venue"])

    X = df[["team1", "team2", "venue", "score1", "wickets1", "balls_left1", "score2", "wickets2", "balls_left2"]].values
    y = df["result"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, team_encoder, venue_encoder), f)

    # Ensure static directory exists
    static_dir = os.path.join("pages", "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Generate Runs vs. Overs Graph
    plt.figure(figsize=(10, 5))
    overs = np.arange(1, 21)
    team1_scores = np.cumsum(np.random.randint(2, 16, 20))
    team2_scores = np.cumsum(np.random.randint(2, 16, 20))

    plt.plot(overs, team1_scores, marker='o', linestyle='-', color='blue', label=team1)
    plt.plot(overs, team2_scores, marker='s', linestyle='-', color='purple', label=team2)

    plt.xlabel("Overs")
    plt.ylabel("Runs")
    plt.title("Runs vs Overs - Line Graph")
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(static_dir, "r_O.png")
    plt.savefig(graph_path, bbox_inches="tight")  
    plt.close()

    print(f"Graph saved at: {graph_path}")  # Debugging Line

    return model

def predict_match(team1, team2, venue, present_score, wickets_left, balls_remaining):
    if not os.path.exists(MODEL_PATH):
        model = train_model()
        if model is None:
            return "Model training failed. No data available."
    else:
        print(team2)
        model = train_model(team1,team2)
        with open(MODEL_PATH, "rb") as f:
            model,team_encoder, venue_encoder = pickle.load(f)

    team1_encoded = team_encoder.transform([team1])[0] if team1 in team_encoder.classes_ else -1
    team2_encoded = team_encoder.transform([team2])[0] if team2 in team_encoder.classes_ else -1
    venue_encoded = venue_encoder.transform([venue])[0] if venue in venue_encoder.classes_ else -1

    if team1_encoded == -1 or team2_encoded == -1:
        team1_encoded = np.random.choice(team_encoder.transform(team_encoder.classes_))
        team2_encoded = np.random.choice(team_encoder.transform(team_encoder.classes_))
    
    if venue_encoded == -1:
        venue_encoded = np.random.choice(venue_encoder.transform(venue_encoder.classes_))

    input_data = [[
        team1_encoded, team2_encoded, venue_encoded,
        present_score, wickets_left, balls_remaining,
        0, 0, 0 
    ]]

    prediction = model.predict(input_data)[0]
    
    return f"{team1} is predicted to win." if prediction == 1 else f"{team2} is predicted to win." if prediction == -1 else "Match is predicted to be a draw."


