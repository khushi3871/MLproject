import sys
from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

# Initialize Flask app
app = Flask(__name__)

# Home page route (shows form)
@app.route('/')
def index():
    return render_template('index.html')


# Prediction route (handles GET and POST)
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            # If user visits /predictdata directly, show the form
            return render_template('home.html')

        # POST request: collect form data and predict
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            math_score=float(request.form.get('math_score')),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert input to DataFrame
        input_df = data.get_data_frame()
        print("Input DataFrame:\n", input_df)

        # Initialize pipeline and predict
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(input_df)

        # Render the same HTML with prediction
        return render_template('home.html', prediction_text=f'Predicted Exam Performance: {results[0]:.2f}')

    except Exception as e:
        # Show error on page
        return render_template('home.html', prediction_text=f"Error occurred: {e}")


# Run app
if __name__ == '__main__':
    app.run(host="0.0.0.0")