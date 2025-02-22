from django.db import models

# Create your models here.
class Team(models.Model):
    name = models.CharField(max_length=100, unique=True)  # Unique team names

    def __str__(self):
        return self.name

class Player(models.Model):
    name = models.CharField(max_length=100)
    team_name = models.CharField(max_length=100)  # This is a string field, NOT a ForeignKey

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


class Predict_winner(models.Model):
    team1 = models.CharField(max_length=100)
    team2 = models.CharField(max_length=100)
    venue = models.CharField(max_length=100)  # 🏟 Venue (affects scores)
    score1 = models.IntegerField()  # 🏏 Runs scored by Team 1
    score2 = models.IntegerField()  # 🏏 Runs scored by Team 2
    wickets1 = models.IntegerField()  # ⚾ Wickets lost by Team 1
    wickets2 = models.IntegerField()  # ⚾ Wickets lost by Team 2
    balls_left1 = models.IntegerField()  # 🎯 Balls remaining for Team 1
    balls_left2 = models.IntegerField()  # 🎯 Balls remaining for Team 2
    winner = models.CharField(max_length=100)  # 🏆 Winning team