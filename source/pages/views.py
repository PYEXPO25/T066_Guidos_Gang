from django.shortcuts import render,redirect # type: ignore
import random
from .models import Player
import joblib # type: ignore
import numpy as np # type: ignore
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
    """Handles player selection and calculates overall ratings for both teams."""
    if request.method == "POST":
        team1 = request.POST.get("team1")
        team2 = request.POST.get("team2")

        # Get selected players from the form
        team1_players = [request.POST.get(f"team1_player{i}") for i in range(1, 12)]
        team2_players = [request.POST.get(f"team2_player{i}") for i in range(1, 12)]

        # Fetch players from the database
        team1_objs = Player.objects.filter(name__in=team1_players)
        team2_objs = Player.objects.filter(name__in=team2_players)

        # Calculate team ratings
        team1_rating = round(sum(player.O_R for player in team1_objs) / 11, 2) if team1_objs else 0.0
        team2_rating = round(sum(player.O_R for player in team2_objs) / 11, 2) if team2_objs else 0.0

        return render(request, "prediction.html", {
            "team1": team1,
            "team2": team2,
            "team1_rating": team1_rating,
            "team2_rating": team2_rating,
            "show_players": False,
            "show_batting_selection": True,
            "batting_first": None,
        })

    return render(request, "home.html", {"show_players": False})


def batting_selection(request):
    """Handles selection of batting first team and renders the appropriate result page."""
    if request.method == "POST":
        batting_order = request.POST.get("batting_order")
        batting_first = request.POST.get("batting_first")
        team1 = request.POST.get("team1")
        team2 = request.POST.get("team2")

        

        # Automatically assign the second team
        batting_second = team2 if batting_first == team1 else team1
        
        if batting_order == "batting_first":  # This should now always have a value
            if batting_first=="":
                batting_first = team1 if batting_second == team1 else team1
                batting_second = team2 if batting_second == team1 else team2
                print("📌 Debugging:")
                print("Batting Order:", batting_order)
                print("Batting First:", batting_first)
                print("Batting Second:", batting_second)  # This should ow always have a value

                return render(request, "result.html", {"batting_first": batting_first, "batting_second": batting_second})
            else:
                batting_first = team2 if batting_second == team2 else team1
                batting_second = team1
                print("Batting First:", batting_first)
                print("Batting Second:", batting_second)
                return render(request, "result.html", {"batting_first": batting_first, "batting_second": batting_second})
            
        elif batting_order == "batting_second":
            return render(request, "result1.html", {"batting_first": batting_first, "batting_second": batting_second})

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

        
        print("Batting First:", team1)
        print("Batting Second:", team2)

        prediction = predict_match(team1, team2, venue, present_score, wickets_left, balls_remaining)
        return render(request, "eee1.html", {"prediction": prediction})
    return render(request, "result1.html")
