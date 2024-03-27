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
model = load_model("../train/weather_prediction_model.h5")
scaler = joblib.load('../train/scaler.gz')

# Load and combine datasets
file_paths = [
    '../dataset/modified_London3.csv',
    '../dataset/modified_London2.csv',
    '../dataset/London13.csv'
]
dfs = [pd.read_csv(file_path) for file_path in file_paths]
combined_df = pd.concat(dfs, ignore_index=True)

# Set datetime as index
combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
combined_df.set_index('datetime', inplace=True)

# Data cleaning and preparation
combined_df = combined_df.loc[~combined_df.index.duplicated()]
combined_df.sort_index(inplace=True)
combined_df = combined_df.interpolate(method='linear')
combined_df['month_day'] = combined_df.index.strftime('%m-%d')

# Select relevant features
features = ['month_day', 'cloudcover', 'tempmax', 'tempmin', 'precip', 'sealevelpressure', 'snowdepth', 'solarradiation']
df_selected = combined_df[features]
df_selected = pd.get_dummies(df_selected, columns=['month_day'])

# Set reference days (look_back) to 5
look_back = 4
num_features = df_selected.shape[1]


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

def is_weather_related(input_text):
    keywords = ['weather', 'cloud cover', 'temperature', 'precipitation', 'sea level pressure', 'snow depth', 'solar radiation', 'max temperature', 'min temperature','sealevel pressure']
    return any(keyword in input_text.lower() for keyword in keywords)


def predict_weather(input_date):
    # Get the last available date in the dataset
    last_available_date = df_selected.index.max().date()

    # If input date is within the dataset range
    if input_date <= last_available_date:
        # Generate dates for previous reference days
        prev_dates = [input_date - timedelta(days=i) for i in range(1, look_back + 1)]

        # Filter the selected dataframe for these dates and features
        prev_data_df = df_selected[df_selected.index.isin(prev_dates)]

        # Scale and reshape the input data
        input_data_scaled = scaler.transform(prev_data_df)
        input_data_reshaped = np.reshape(input_data_scaled, (1, look_back, num_features))

        # Make predictions
        prediction = model.predict(input_data_reshaped)
        prediction_inversed = scaler.inverse_transform(prediction)

        return prediction_inversed[0]

    # If input date is beyond the dataset range
    # In this case the weather is found for the same dates in the last two years.
    # features for the first four days of both dates are averaged and fit into the model.
    else:
        # Extract month and day from input date
        month = input_date.month
        day = input_date.day

        # Initialize an empty DataFrame to store the last two years data
        last_two_years_data = pd.DataFrame()

        # Start from the previous year and go back until we find two years with data
        year = input_date.year - 1
        while len(last_two_years_data) < 2 and year >= df_selected.index.min().year:
            same_month_day_data = df_selected[
                (df_selected.index.month == month) & (df_selected.index.day == day) & (df_selected.index.year == year)]
            if len(same_month_day_data) > 0:
                last_two_years_data = pd.concat([last_two_years_data, same_month_day_data])
            year -= 1

        # Check if we have enough data
        if len(last_two_years_data) < 2:
            return "Insufficient historical data for the specified date."

        # Generate dates for previous reference days for each year
        prev_dates_list = []
        for date in last_two_years_data.index:
            prev_dates = [date - timedelta(days=i) for i in range(look_back)]
            prev_dates_list.extend(prev_dates)

        # Filter the selected dataframe for these dates and features
        prev_data_df = df_selected[df_selected.index.isin(prev_dates_list)]

        # Group by month and day and calculate the mean
        mean_data = prev_data_df.groupby([prev_data_df.index.month, prev_data_df.index.day]).mean()

        # Scale and reshape the input data
        input_data_scaled = scaler.transform(mean_data)
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
        if not is_weather_related(data):
            response = "Sorry, I can only deal with weather prediction. Please ask about cloud cover, temperature, precipitation, sea level pressure, snow depth, or solar radiation."
        else:

            if input_date is None:
                response = "Invalid date format. Please provide a valid date."
            else:
                # Predict weather
                prediction = predict_weather(input_date)
                if isinstance(prediction, str):
                    response = prediction
                    print(response)
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
                        if float(prediction[3]) <= 0:
                            response = "No rain"
                        else:
                            response = f"Predicted Precipitation for {input_date}: {prediction[3]:.2f}"
                    elif 'sea level pressure' in data.lower() or 'sealevel pressure' in data.lower():
                        response = f"Predicted Sea Level Pressure for {input_date}: {prediction[4]:.2f}"
                    elif 'snow depth' in data.lower():
                        if float(prediction[5]) <= 0:
                            response = "No Snow"
                        else:
                            response = f"Predicted Snow Depth for {input_date}: {prediction[5]:.2f}"
                    elif 'solar radiation' in data.lower():
                        response = f"Predicted Solar Radiation for {input_date}: {prediction[6]:.2f}"
                    elif 'weather' in data.lower():
                        if float(prediction[3]) <= 0:
                            rain = "No rain"
                        else:
                            rain = round(float(prediction[3]),2)
                        if float(prediction[5]) <= 0:
                            snow = "No Snow"
                        else:
                            snow = round(float(prediction[5]), 2)
                        response = f"Weather prediction for {input_date}:\n" \
                                   f"Cloud Cover: {prediction[0]:.2f}\n" \
                                   f"Maximum Temperature: {prediction[1]:.2f}\n" \
                                   f"Minimum Temperature: {prediction[2]:.2f}\n" \
                                   f"Precipitation: {rain}\n" \
                                   f"Sea Level Pressure: {prediction[4]:.2f}\n" \
                                   f"Snow Depth: {snow}\n" \
                                   f"Solar Radiation: {prediction[6]:.2f}"
                    else:
                        response = "Sorry, I can only deal with weather prediction. Please ask about cloud cover, temperature, precipitation, sea level pressure, snow depth, solar radiation, or weather for overall prediction"

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