from django.shortcuts import render,redirect # type: ignore
import random
from pages.models import Player,Ipl_matches
import joblib # type: ignore
import numpy as np # type: ignore
from pages.train1 import predict_match
from pages.models import OverallResult,GroundInfo
from django.shortcuts import render
from collections import Counter
from collections import OrderedDict,defaultdict
from django.http import HttpResponseRedirect
from django.urls import reverse
import os
def home(request):
    return render(request, 'home.html')

def dc(request):
    return render(request, 'dc.html')

def csk(request):
    return render(request, 'csk.html')

def gt(request):
    return render(request, 'gt.html')


def kkr(request):
    return render(request, 'kkr.html')


def lsg(request):
    return render(request, 'lsg.html')


def mi(request):
    return render(request, 'mi.html')


def pbks(request):
    return render(request, 'pbks.html')


def rcb(request):
    return render(request, 'rcb.html')

def rr(request):
    return render(request, 'rr.html')

def srh(request):
    return render(request, 'srh.html')

def prediction_view(request):
    return render(request, 'prediction.html')

def predict(request):
    # Your prediction logic here
    return render(request, 'result.html')
def contact_us(request):
    return render(request, 'contact_us.html')


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
        model = joblib.load("D:/raahulkanna/New Desktop/django12/T066_Guidos_Gang/source/pages/prediction/ml_model.pkl")

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


def predict_win(request):
    if request.method == "POST":
        team1 = request.POST.get("batting_first")
        team2 = request.POST.get("batting_second")
        venue = request.POST["venue"]
        present_score = int(request.POST["present_score"])
        wickets_left = int(request.POST["wickets_left"])
        balls_remaining = int(request.POST["balls_remaining"])
        target = int(request.POST["target"])
        bat_first = team1

        print("Batting First:", team1)
        print("Batting Second:", team2)
        print("venue:", venue)
        print("pre:", present_score)
        print("wic:", wickets_left)
        print("bal:", balls_remaining)
        print("tar:", target)
        print("Bat first:", bat_first)

        # Lowercase for model input
        team1_lower = team1.lower()
        team2_lower = team2.lower()
        bat_first_lower = bat_first.lower()

        # 🎯 Make Prediction
        prediction = predict_match(
            team1_lower,
            team2_lower,
            venue,
            present_score,
            wickets_left,
            balls_remaining,
            bat_first_lower,
            target,
        )

        # 📈 Calculate Key Factors
        runs_needed = target - present_score
        balls_used = 120 - balls_remaining
        curr_rr = round(present_score / (balls_used / 6), 1) if balls_used != 0 else 0
        req_rr = round(runs_needed / (balls_remaining / 6), 1) if balls_remaining != 0 else 0

        # Determine chasing team (not bat_first)
        chasing_team = team2 if bat_first == team1 else team1
        wickets_lost = 10 - wickets_left

        # Format key factors explanation
        key_factors = (
            f"📈 Key Factors Influencing Prediction:\n"
            f"• {chasing_team.upper()} has lost {wickets_lost} wickets\n"
            f"• Only {runs_needed} runs needed in {balls_remaining} balls\n"
            f"• Current Run Rate: {curr_rr} (Required: {req_rr})"
        )

        context = {
        "prediction": prediction,
        "team1": team2.upper(), # If you're still using it
        "chasing_team": chasing_team.upper(),
        "wickets_left": 10 - wickets_left,
        "balls_remaining": balls_remaining,
        "runs_needed": runs_needed,
        "curr_rr": curr_rr,
        "req_rr": req_rr,
}


        return render(request, "eee1.html", context)

    return render(request, "result1.html")

def clean_overallresult_duplicates():
    grouped = defaultdict(list)
    for obj in OverallResult.objects.all():
        grouped[obj.match_number].append(obj)

    for match_number, objects in grouped.items():
        if len(objects) > 1:
            # keep latest (highest id), delete the rest
            sorted_objs = sorted(objects, key=lambda x: x.id, reverse=True)
            for obj in sorted_objs[1:]:
                obj.delete()
def overall_winner(request):
    if request.method == "POST":
        for i in range(1, 11):  # Loop through 10 matches
            team1 = request.POST.get(f"team1_{i}")
            team2 = request.POST.get(f"team2_{i}")
            venue = request.POST.get(f"venue_{i}")
            pitch_type = request.POST.get(f"pitch_type_{i}")
            predicted_winner = request.POST.get(f"predicted_winner_{i}")

            if team1 and team2 and team1 != team2 and predicted_winner:
                OverallResult.objects.create(
                    match_number=i,
                    team1=team1,
                    team2=team2,
                    venue=venue,
                    pitch_type=pitch_type,
                    predicted_winner=predicted_winner
                )
        return redirect("overall_result")

    # Team and ground info
    teams = ["CSK", "RCB", "MI", "GT", "RR", "DC", "KKR", "LSG"]
    pitch_types = list(GroundInfo.objects.values_list("pitch_type", flat=True).distinct())
    venues = list(GroundInfo.objects.values_list("venue", flat=True).distinct())

    # Create random values for pre-fill observation
    match_defaults = []
    for i in range(1, 11):
        team1, team2 = random.sample(teams, 2)
        venue = random.choice(venues) if venues else "Unknown"
        pitch_type = random.choice(pitch_types) if pitch_types else "Balanced"
        predicted_winner = random.choice([team1, team2])
        match_defaults.append({
            "match_number": i,
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "pitch_type": pitch_type,
            "predicted_winner": predicted_winner,
        })

    return render(request, "overall_predict.html", {
        "teams": teams,
        "venues": venues,
        "pitch_types": pitch_types,
        "match_defaults": match_defaults
    })

def overall_result(request):
    clean_overallresult_duplicates()
    results = OverallResult.objects.all().order_by("match_number")
    winner_list = [r.predicted_winner for r in results]
    winner_counts = dict(Counter(winner_list))

    if winner_counts:
        max_win = max(winner_counts.values())
        top_teams = [team for team, count in winner_counts.items() if count == max_win]

        if len(top_teams) == 1:
            overall_winner = top_teams[0]
            tie = False
        else:
            overall_winner = None
            tie = True
    else:
        overall_winner = None
        tie = False
        top_teams = []

    return render(request, "overall_result.html", {
        "results": results,
        "winner_counts": winner_counts,
        "overall_winner": overall_winner,
        "tie": tie,
        "top_teams": top_teams,
    })

def final_matchup(request):
    if request.method == "POST":
        tied_teams = request.POST.getlist("team")
        final_winner = random.choice(tied_teams)

        # You can store this result in session or DB if needed
        request.session["final_winner"] = final_winner
        return HttpResponseRedirect(reverse("final_result"))
    
def final_result(request):
    winner = request.session.get("final_winner", None)
    return render(request, "final_result.html", {"final_winner": winner})

MODEL_PATH = os.path.join("pages", "prediction", "player_predict_model.pkl")
ENCODERS_PATH = os.path.join("pages", "prediction", "player_predict_encoders.pkl")

# 🔒 Safe model/encoder loading
try:
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
except Exception as e:
    model = None
    label_encoders = None
    print(f"❌ Failed to load model or encoders: {e}")

def predict_player(request):
    if request.method == "POST":
        try:
            if not model or not label_encoders:
                return render(request, "predict_player.html", {"error": "Model or encoders not loaded. Please retrain the model."})

            # 📥 Get form inputs
            strike_rate = float(request.POST.get("strike_rate"))
            runs = int(request.POST.get("runs"))
            wickets = int(request.POST.get("wickets"))
            overs = float(request.POST.get("overs"))
            venue = request.POST.get("venue")
            pitch_type = request.POST.get("pitch_type")
            economy = int(request.POST.get("economy"))

            # 🎯 Encode categorical fields
            venue_encoded = label_encoders["venue"].transform([venue])[0]
            pitch_encoded = label_encoders["pitch_type"].transform([pitch_type])[0]

            # 🧾 Prepare input
            input_data = np.array([[strike_rate, runs, wickets, overs, venue_encoded, pitch_encoded, economy]])

            # 🤖 Predict and decode player
            prediction_encoded = model.predict(input_data)[0]
            player_name = label_encoders["player_name"].inverse_transform([prediction_encoded])[0]

            return render(request, "predict_player.html", {
                "player_name": player_name,
                "strike_rate": strike_rate,
                "runs": runs,
                "wickets": wickets,
                "overs": overs,
                "venue": venue,
                "pitch_type": pitch_type,
                "economy": economy,
            })

        except Exception as e:
            return render(request, "predict_player.html", {"error": f"⚠️ Error: {str(e)}"})

    return render(request, "predict_player.html")