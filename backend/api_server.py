# backend/api_server.py
# running local on localhost:5000
# /api/sleepiness will return live score.
from flask import Flask, jsonify
from flask_cors import CORS 
from threading import Lock

app = Flask(__name__)
CORS(app)
sleepiness_data = {
    "sleepiness_score": 100,
    "classification": "Open",
    "timestamp": "N/A"
}
lock = Lock()

@app.route('/api/sleepiness', methods=['GET'])
def get_sleepiness():
    with lock:
        return jsonify(sleepiness_data)

def update_sleepiness(score, classification, timestamp):
    with lock:
        sleepiness_data["sleepiness_score"] = score
        sleepiness_data["classification"] = classification
        sleepiness_data["timestamp"] = timestamp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
