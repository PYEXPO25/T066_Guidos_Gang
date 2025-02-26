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
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from pages.models import Predict_win

MODEL_PATH = os.path.join("pages", "prediction", "rf_model.pkl")

def train_model(team1, team2):
    print("🚀 train_model() function started!")  # Debugging
    print(team2)

    matches = Predict_win.objects.all()
    if not matches:
        return None
    
    data = []
    team_names = set()
    venue_names = set()
    bat_first_names = set()
    winner_names = set()
    
    for match in matches:
        team_names.update([match.team1, match.team2])
        venue_names.add(match.venue)
        bat_first_names.add(match.bat_first)
        winner_names.add(match.winner)
        data.append([
    match.team1, match.team2, match.venue,
    match.pre_score, match.target, match.balls_rem,
    match.wic_left, match.bat_first, match.winner  # ✅ Keep `match.winner` as a team name
])


    df = pd.DataFrame(data, columns=[
    "team1", "team2", "venue",
    "pre_score", "target", "balls_rem",
    "wic_left", "bat_first", "winner"
])

    df.drop(columns=['id'], errors='ignore', inplace=True)  # Now it is safe to use

    # Encoding
    team_encoder = LabelEncoder()
    venue_encoder = LabelEncoder()
    bat_first_encoder = LabelEncoder()
    winner_encoder = LabelEncoder() 
    team_encoder.fit(list(team_names))
    venue_encoder.fit(list(venue_names))
    bat_first_encoder.fit(list(bat_first_names))
    winner_encoder.fit(list(winner_names))

    df["team1"] = team_encoder.transform(df["team1"])
    df["team2"] = team_encoder.transform(df["team2"])
    df["venue"] = venue_encoder.transform(df["venue"])
    df["bat_first"]= bat_first_encoder.transform(df["bat_first"])
    df["winner"] = winner_encoder.transform(df["winner"]) 

    X = df[["team1","team2","venue","pre_score", "target", "balls_rem","wic_left","bat_first"]].values
    y = df["winner"]

    print("Unique classes in dataset:", np.unique(y))  # y contains match results
    print("Number of classes:", len(np.unique(y)))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y, y_pred, average='weighted'))

    print("Team Encoder Classes:", team_encoder.classes_)
    print("Venue Encoder Classes:", venue_encoder.classes_)
    print("Bat First Encoder Classes:", bat_first_encoder.classes_)
    print("Winner Encoder Classes:", winner_encoder.classes_)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, team_encoder, venue_encoder, bat_first_encoder,winner_encoder), f)

    # Ensure static directory exists
    static_dir = os.path.join("pages", "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Generate Runs vs. Overs Graph
    plt.figure(figsize=(10, 5))
    overs = np.arange(1, 21)
    team1_scores = np.cumsum(np.random.randint(2, 16, 20))
    team2_scores = np.cumsum(np.random.randint(2, 16, 20))
    team1=team1.upper()
    team2=team2.upper()
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

def predict_match(team1, team2, venue, present_score, wickets_left, balls_remaining, bat_first,target):
    if not os.path.exists(MODEL_PATH):
        model = train_model()
        if model is None:
            return "Model training failed. No data available."
    else:
        print(team2)
        model = train_model(team1, team2)
        # Load Model and Encoders
        with open(MODEL_PATH, "rb") as f:
            model, team_encoder, venue_encoder, bat_first_encoder, winner_encoder = pickle.load(f)

    # Encode input using the same encoders
        team1_encoded = team_encoder.transform([team1])[0] if team1 in team_encoder.classes_ else -1
        team2_encoded = team_encoder.transform([team2])[0] if team2 in team_encoder.classes_ else -1
        venue_encoded = venue_encoder.transform([venue])[0] if venue in venue_encoder.classes_ else -1
        if venue_encoded == -1:
            return f"Error: Venue '{venue}' not found in training data. Retrain with updated data."

        bat_first_encoded = bat_first_encoder.transform([bat_first])[0] if bat_first in bat_first_encoder.classes_ else -1
        print(team1_encoded)
        print(team2_encoded)
        print(venue_encoded)
        print(bat_first_encoded)
    # Ensure no -1 values (meaning unseen data)
    
    # Prepare input array
        input_data = np.array([[team1_encoded, team2_encoded, venue_encoded, present_score, target, balls_remaining, wickets_left, bat_first_encoded]])

    # Make prediction
        prediction = model.predict(input_data)[0]
        print("pre",prediction)
        predicted_winner = winner_encoder.inverse_transform([prediction])[0]
        predicted_winner=predicted_winner.upper()
  # Convert back to team name
        print(predicted_winner)
        return f"Predicted Winner: {predicted_winner}"

a=predict_match('rcb','mi','chinnaswamy Stadium',101,2,45,'mi',202)
