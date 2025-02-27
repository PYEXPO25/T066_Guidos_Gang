import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_runs(score, wickets_left, target, is_chasing=False, team_name="Generic"):
    overs = np.arange(1, 21)  # 20 Overs
    runs_per_over = []
    current_score = score
    momentum = 1.0  # Track momentum (higher = better form)
    
    # Team-specific aggression levels
    team_aggression = {
        "MI": 1.2, "CSK": 1.1, "RCB": 1.3, "SRH": 1.0, "KKR": 1.25, 
        "RR": 1.05, "PBKS": 1.15, "DC": 1.1, "GT": 1.15, "LSG": 1.2
    }
    aggression = team_aggression.get(team_name.upper(), 1.0)

    for over in overs:
        if wickets_left <= 0:  # Stop scoring if all wickets lost
            runs_per_over.append(current_score)
            continue

        # Dynamic scoring pattern based on match phase
        if over <= 6:  # Powerplay (Explosive)
            base_run_rate = np.random.randint(5, 14)
        elif over <= 15:  # Middle Overs (Steady)
            base_run_rate = np.random.randint(3, 11)
        else:  # Death Overs (High Risk)
            base_run_rate = np.random.randint(6, 20) if wickets_left > 2 else np.random.randint(2, 12)

        # Additional variation for realistic ups & downs
        run_variation = np.random.randint(-5, 6)  # More extreme fluctuations

        # Chasing team logic
        if is_chasing:
            required_run_rate = (target - current_score) / max(20 - over, 1)

            if required_run_rate > base_run_rate:  # Need acceleration
                base_run_rate = int(required_run_rate * 1.2) + np.random.randint(1, 4)
            elif required_run_rate < base_run_rate:  # Playing safe
                base_run_rate = int(required_run_rate * 0.85)

            runs = max(0, min(int(base_run_rate * aggression * momentum + run_variation), 26))
        else:
            runs = max(0, min(int(base_run_rate * aggression * momentum + run_variation), 24))

        current_score += runs
        runs_per_over.append(current_score)

        # Wickets Falling (Higher Risk in Death Overs)
        wicket_chance = 0.2 if over < 15 else 0.35  
        if np.random.rand() < wicket_chance:
            wickets_left -= 1
            aggression *= 0.85  # Reduce aggression after wickets
            momentum *= np.random.uniform(0.7, 0.95)  # Losing wickets kills momentum

        # Random Momentum Swings (Hot & Cold Phases)
        if np.random.rand() < 0.15:  # 15% chance of form boost/drop
            momentum *= np.random.uniform(0.7, 1.3)

    return np.array(runs_per_over)

def generate_graph(team1, team2, present_score, wickets_left, target):
    static_dir = os.path.join("pages", "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    overs = np.arange(1, 21)  # 20 overs
    team1_scores = simulate_runs(present_score, wickets_left, target, is_chasing=False, team_name=team1)
    team2_scores = simulate_runs(present_score, wickets_left, target, is_chasing=True, team_name=team2)

    plt.figure(figsize=(12, 6))
    team1 = team1.upper()
    team2 = team2.upper()
    plt.plot(overs, team1_scores, marker='o', linestyle='-', color='blue', label=f"{team1} (Bat First)")
    plt.plot(overs, team2_scores, marker='s', linestyle='-', color='red', label=f"{team2} (Chasing)")

    plt.xlabel("Overs")
    plt.ylabel("Runs")
    plt.title(f"Runs vs Overs - {team1} vs {team2}")
    plt.xticks(overs)
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(static_dir, "r_O.png")
    plt.savefig(graph_path, bbox_inches="tight")
    plt.close()

    print(f"Graph saved at: {graph_path}")  # Debugging Line

# Example call:
generate_graph("RCB", "MI", 101, 7, 202)
