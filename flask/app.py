from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("iris_classification_model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

# Define route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from request
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)  # Expecting list of features

        # Predict
        prediction = model.predict(features)
        if prediction[0] == 0:
            predicted = "setosa"
        elif prediction[0] == 1:
            predicted = "versicolor"
        elif prediction[0] == 2:
            predicted = "virginica"
       
        # Return response
        return jsonify({"predicted_class": predicted}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
