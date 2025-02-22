document.addEventListener('click', function () {
    const audio = document.getElementById('background-music');
    audio.play().catch(error => {
        console.log('Playback failed:', error);
    });
}, { once: true });

let selectedTeam = "";

document.getElementById('select-team-button').addEventListener('click', () => {
    selectedTeam = document.getElementById('team').value;
    if (selectedTeam) {
        document.getElementById('selected-team').innerText = `You selected ${selectedTeam} to bat first!`;
    } else {
        document.getElementById('selected-team').innerText = "Please select a team.";
    }
});

document.getElementById('prediction-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const target = parseInt(document.getElementById('target').value);
    const score = parseInt(document.getElementById('present-score').value);
    const wickets = parseInt(document.getElementById('wickets-down').value);
    const balls = parseInt(document.getElementById('balls-left').value);

    if (target && score >= 0 && wickets >= 0 && balls >= 0) {
        const chance = Math.min(100, Math.max(0, ((score / target) * 100 - (wickets * 5) + (balls / 6)).toFixed(2)));
        document.getElementById('final-prediction').innerText = `${selectedTeam} has a winning chance of: ${chance}%`;
    } else {
        document.getElementById('final-prediction').innerText = "Please fill all fields correctly.";
    }
});
