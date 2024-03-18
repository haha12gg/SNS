import socket
from datetime import datetime, timedelta
import re
from _thread import *
import threading
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

print_lock = threading.Lock()

# Load the trained model and scaler
model = load_model("weather_prediction_model.h5")
scaler = joblib.load('scaler.gz')

# Load and combine datasets
file_paths = [
    'modified_London3.csv',
    'modified_London2.csv',
    'London13.csv'
]
dfs = [pd.read_csv(file_path) for file_path in file_paths]
combined_df = pd.concat(dfs, ignore_index=True)

# Data cleaning and preparation
combined_df.drop_duplicates(subset=['datetime'], inplace=True)
combined_df.sort_values('datetime', inplace=True)
combined_df = combined_df.interpolate(method='linear')
combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
combined_df['month_day'] = combined_df['datetime'].dt.strftime('%m-%d')

# Select relevant features
features = ['datetime', 'month_day', 'cloudcover', 'tempmax', 'tempmin', 'precip', 'sealevelpressure', 'snowdepth', 'solarradiation']
df_selected = combined_df[features]
df_selected = pd.get_dummies(df_selected, columns=['month_day'])

# Set reference days (look_back) to 5
look_back = 4
num_features = df_selected.drop(columns=['datetime']).shape[1]

# Convert data to particular format eg tomorrow to
def convert_date_format(input_text):
    today = datetime.now().date()
    if 'tomorrow' in input_text:
        return today + timedelta(days=1)
    elif 'today' in input_text:
        return today
    else:
        match = re.search(r'\b\d{1,2}\s+\w+\s+\d{4}\b', input_text)
        if match:
            date_str = match.group(0)
            try:
                return datetime.strptime(date_str, '%d %B %Y').date()
            except ValueError:
                try:
                    return datetime.strptime(date_str, '%d %b %Y').date()
                except ValueError:
                    pass
    return None

def predict_weather(input_date):
    # Generate dates for previous reference days
    prev_dates = [input_date - timedelta(days=i) for i in range(1, look_back + 1)]

    # Filter the selected dataframe for these dates and features
    prev_data_df = df_selected[df_selected['datetime'].isin(prev_dates)].drop(columns=['datetime'])

    # Check if we have enough data
    if len(prev_data_df) < look_back:
        return "Insufficient historical data for the specified date."

    # Scale and reshape the input data
    input_data_scaled = scaler.transform(prev_data_df)
    input_data_reshaped = np.reshape(input_data_scaled, (1, look_back, num_features))

    # Make predictions
    prediction = model.predict(input_data_reshaped)
    prediction_inversed = scaler.inverse_transform(prediction)

    return prediction_inversed[0]

# thread function
def threaded(c):
    while True:
        # data received from client
        data = c.recv(1024).decode('utf-8')
        if not data:
            print('Bye')
            print_lock.release()
            break

        # Convert input to date
        input_date = convert_date_format(data)
        if input_date is None:
            response = "Invalid date format. Please provide a valid date."
        else:
            # Predict weather
            prediction = predict_weather(input_date)
            if isinstance(prediction, str):
                response = prediction
            else:
                if 'cloud cover' in data.lower():
                    response = f"Predicted Cloud Cover for {input_date}: {prediction[0]:.2f}"
                elif 'temperature' in data.lower():
                    response = f"Predicted Maximum Temperature for {input_date}: {prediction[1]:.2f}, Predicted Minimum Temperature: {prediction[2]:.2f}"
                elif 'max temperature' in data.lower():
                    response = f"Predicted Maximum Temperature for {input_date}: {prediction[1]:.2f}"
                elif 'min temperature' in data.lower():
                    response = f"Predicted Minimum Temperature for {input_date}: {prediction[2]:.2f}"
                elif 'precipitation' in data.lower():
                    response = f"Predicted Precipitation for {input_date}: {prediction[3]:.2f}"
                elif 'sea level pressure' in data.lower() or 'sealevel pressure' in data.lower():
                    response = f"Predicted Sea Level Pressure for {input_date}: {prediction[4]:.2f}"
                elif 'snow depth' in data.lower():
                    response = f"Predicted Snow Depth for {input_date}: {prediction[5]:.2f}"
                elif 'solar radiation' in data.lower():
                    response = f"Predicted Solar Radiation for {input_date}: {prediction[6]:.2f}"
                else:
                    response = f"Weather prediction for {input_date}:\n" \
                               f"Cloud Cover: {prediction[0]:.2f}\n" \
                               f"Maximum Temperature: {prediction[1]:.2f}\n" \
                               f"Minimum Temperature: {prediction[2]:.2f}\n" \
                               f"Precipitation: {prediction[3]:.2f}\n" \
                               f"Sea Level Pressure: {prediction[4]:.2f}\n" \
                               f"Snow Depth: {prediction[5]:.2f}\n" \
                               f"Solar Radiation: {prediction[6]:.2f}"

        # send back response to client
        c.send(response.encode('utf-8'))

    # connection closed
    c.close()

def Main():
    host = ""
    port = 12345
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    print("socket binded to port", port)

    s.listen(5)
    print("socket is listening")

    while True:
        c, addr = s.accept()
        print_lock.acquire()
        print('Connected to :', addr[0], ':', addr[1])
        start_new_thread(threaded, (c,))
    s.close()

if __name__ == '__main__':
    Main()