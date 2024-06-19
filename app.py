from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# Load the trained models
temp_model = joblib.load('temp_model.pkl')
humidity_model = joblib.load('humidity_model.pkl')

# Define time intervals for the next day
time_intervals = ['6:00 PM', '6:30 PM', '7:00 PM', '7:30 PM', '8:00 PM', '8:30 PM', '9:00 PM', '9:30 PM',
                  '10:00 PM', '10:30 PM', '11:00 PM', '12:30 AM', '1:00 AM', '1:30 AM', '2:00 AM',
                  '2:30 AM', '3:00 AM', '3:30 AM', '4:00 AM', '4:30 AM', '5:00 AM', '5:30 AM', '6:00 AM',
                  '6:30 AM', '7:00 AM', '7:30 AM', '8:00 AM', '8:30 AM', '9:00 AM', '9:30 AM', '10:00 AM',
                  '10:30 AM', '11:00 AM', '11:30 AM', '12:00 PM', '12:30 PM', '1:00 PM', '1:30 PM', '2:00 PM',
                  '2:30 PM', '3:00 PM', '3:30 PM', '4:00 PM', '4:30 PM', '5:00 PM', '5:30 PM']

# Generate the time points for the next day
next_day_times = [datetime.strptime(time, '%I:%M %p') + timedelta(days=1) for time in time_intervals]

@app.route('/')
def index():
    return render_template('index.html')  # Replace 'your_html_file.html' with the name of your HTML file

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from AJAX request
    data = request.json
    
    # Predict temperature and humidity for all time frames
    predictions = {}
    for time in next_day_times:
        user_data = pd.DataFrame({'Hour': [time.hour], 'Minute': [time.minute]})
        temp_prediction = temp_model.predict(user_data)[0]
        humidity_prediction = humidity_model.predict(user_data)[0]
        predictions[time.strftime('%I:%M %p')] = {'temperature': temp_prediction, 'humidity': humidity_prediction}
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
