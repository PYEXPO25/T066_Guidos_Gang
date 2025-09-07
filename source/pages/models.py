
from django.db import models

# Create your models here.
class Team(models.Model):
    name = models.CharField(max_length=100, unique=True)
      # Unique team names

    def __str__(self):
        return self.name

class Player(models.Model):
    name = models.CharField(max_length=100)
    team_name = models.CharField(max_length=100)
    O_R = models.FloatField(default=0.0)  # This is a string field, NOT a ForeignKey

    def __str__(self):
        return self.name
    
class BattingFirstPredict(models.Model):
    team1 = models.CharField(max_length=50)
    team2 = models.CharField(max_length=50)
    batting_first = models.CharField(max_length=50)
    venue = models.CharField(max_length=100)
    present_score = models.IntegerField()
    wickets_left = models.IntegerField()
    balls_remaining = models.IntegerField()
    predict_target = models.IntegerField()

    def __str__(self):
        return f"{self.batting_first} vs {self.team2} at {self.venue}"


class Predict_win(models.Model):
    team1 = models.CharField(max_length=100)
    team2 = models.CharField(max_length=100)
    venue = models.CharField(max_length=100)  # 🏟 Venue (affects scores)
    pre_score = models.IntegerField()  # 🏏 Runs scored by Team 1
    bat_first = models.CharField(max_length=30,default='csk')  # 🏏 Runs scored by Team 2
    wic_left = models.IntegerField()  # ⚾ Wickets lost by Team 1
    balls_rem = models.IntegerField()  # ⚾ Wickets lost by Team 2
    target = models.IntegerField()  # 🎯 Balls remaining for Team 1 # 🎯 Balls remaining for Team 2
    winner = models.CharField(max_length=100)  # 🏆 Winning team

class Ipl_matches(models.Model):
    team1 = models.CharField(max_length=100)
    team2 = models.CharField(max_length=100)
    venue = models.CharField(max_length=100)
    date = models.DateField()
    predicted_winner = models.CharField(max_length=100, null=True, blank=True)  # Stores predicted winner

class OverallResult(models.Model):
    match_number = models.IntegerField()       # 🔢 Sequence helps track match flow
    team1 = models.CharField(max_length=50)    # 🟡 Input team
    team2 = models.CharField(max_length=50)    # 🔵 Opponent team
    venue = models.CharField(max_length=100)   # 🏟️ Location-based influence
    pitch_type = models.CharField(max_length=50)  # 🌱 Flat, green, dry, turning
    predicted_winner = models.CharField(max_length=50)  # 🏆 Output label

class GroundInfo(models.Model):
    venue = models.CharField(max_length=100)
    pitch_type = models.CharField(max_length=50)


class PlayerPredict(models.Model):
    player_name = models.CharField(max_length=50)
    
    strike_rate = models.FloatField(null=True, blank=True)
    runs = models.IntegerField(null=True, blank=True)
    
    wickets = models.IntegerField(null=True, blank=True)
    overs = models.FloatField(null=True, blank=True)
    
    venue = models.CharField(max_length=100)
    pitch_type = models.CharField(max_length=50)  # e.g., Dry, Green, Dusty, Balanced
    economy = models.IntegerField(default=0)
    
    def __str__(self):
        return self.player_name