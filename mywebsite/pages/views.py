from django.shortcuts import render,redirect
import random
from .models import Player
import joblib
import numpy as np
from .train1 import predict_match



def home(request):
    return render(request, 'home.html')

def prediction_view(request):
    return render(request, 'prediction.html')

def predict(request):
    # Your prediction logic here
    return render(request, 'result.html')

def select_teams(request):
    """Handles team selection and shows available players."""
    if request.method == "POST":
        team1 = request.POST.get("team1")
        team2 = request.POST.get("team2")

        if team1 and team2 and team1 != team2:
            all_team1_players = list(Player.objects.filter(team_name=team1).values_list("name", flat=True))
            all_team2_players = list(Player.objects.filter(team_name=team2).values_list("name", flat=True))

            # Auto-select 11 players randomly
            team1_players = random.sample(all_team1_players, min(11, len(all_team1_players)))
            team2_players = random.sample(all_team2_players, min(11, len(all_team2_players)))

            return render(request, "prediction.html", {
                "team1": team1,
                "team2": team2,
                "team1_players": team1_players,
                "team2_players": team2_players,
                "all_team1_players": all_team1_players,
                "all_team2_players": all_team2_players,
                "show_players": True,  # ✅ Show player selection first
                "show_batting_selection": False,  # ❌ Hide batting selection initially
                "batting_first": None,  # Ensure no team is set initially
            })

    return render(request, "home.html", {"show_players": False})

def submit_players(request):
    """Handles player selection and moves to batting selection."""
    if request.method == "POST":
        team1 = request.POST.get("team1")
        team2 = request.POST.get("team2")

        team1_players = [request.POST.get(f"team1_player{i}") for i in range(1, 12)]
        team2_players = [request.POST.get(f"team2_player{i}") for i in range(1, 12)]

        return render(request, "prediction.html", {
            "team1": team1,
            "team2": team2,
            "show_players": False,  # ❌ Hide player selection
            "show_batting_selection": True,  # ✅ Show batting selection
            "batting_first": None,  # Ensure no team is set initially
        })

    return render(request, "home.html", {"show_players": False})


def batting_selection(request):
    """Handles selection of batting first team and renders a new page."""
    if request.method == "POST":
        batting_order = request.POST.get("batting_order")
        batting_first = request.POST.get("batting_first")
        batting_second = request.POST.get("batting_second")
        if batting_order == "batting_first":
            return render(request, "result.html", {"batting_first": batting_first})
        elif batting_order == "batting_second":
            return render(request, "result1.html", {"batting_first": batting_first,"batting_second": batting_second})  # Redirect to result1.html for user input

    return redirect("home")

def predict_target(request):
    if request.method == 'POST':
        # Get user inputs from form
        batting_first = request.POST.get("batting_first")
        venue = request.POST.get("venue")
        present_score = int(request.POST.get("present_score"))
        balls_remaining = int(request.POST.get("balls_remaining"))
        wickets_left = int(request.POST.get("wickets_left"))

        # Encode venue
        venue_mapping = {
            "Wankhede Stadium": 1, "Chinnaswamy Stadium": 2, "Eden Gardens": 3,
            "Kotla": 4, "Chepauk": 5, "Arun Jaitley Stadium": 6,
            "MA Chidambaram Stadium": 7, "Rajiv Gandhi International Stadium": 8,
            "Narendra Modi Stadium": 9, "Sawai Mansingh Stadium": 10,
            "BRSABV Ekana Cricket Stadium": 11, "Punjab Cricket Association Stadium": 12
        }
        venue_encoded = venue_mapping.get(venue, 0)

        # Load trained model
        model = joblib.load("C:/Users/raahu/Desktop/Django8/mywebsite/pages/prediction/ml_model.pkl")

        # Predict target score
        input_features = np.array([[present_score, balls_remaining, wickets_left, venue_encoded]])
        predicted_target = int(model.predict(input_features)[0])

        return render(request, "eee.html", {
            "batting_first": batting_first,
            "predicted_target": predicted_target,
            "venue": venue,
            "present_score": present_score,
            "balls_remaining": balls_remaining,
            "wickets_left": wickets_left
        })

    return render(request, "result.html")


def predict_winner(request):
    if request.method == "POST":
        team1 = request.POST.get("batting_first")
        team2 = request.POST.get("batting_second")
        venue = request.POST["venue"]
        present_score = int(request.POST["present_score"])
        wickets_left = int(request.POST["wickets_left"])
        balls_remaining = int(request.POST["balls_remaining"])

        prediction = predict_match(team1, team2, venue, present_score, wickets_left, balls_remaining)
        return render(request, "eee1.html", {"prediction": prediction})
    return render(request, "result1.html")
