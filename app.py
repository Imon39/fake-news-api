from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing text"}), 400
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    return jsonify({"is_fake": bool(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
