import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and scaler
model = load_model("weather_prediction_model.h5")
scaler = joblib.load('scaler.gz')

# Load and combine datasets
file_paths = [
    'London2_with_sunshine.csv',
    'London3_with_sunshine.csv',
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

# Ensure the input sequence matches the model's expected shape
look_back = 10
num_features = df_selected.drop(columns=['datetime']).shape[1]

# Interactive prediction
while True:
    user_input = input("Enter a date (YYYY-MM-DD) for weather prediction or 'exit' to quit: ")
    if user_input.lower() == 'exit':
        break

    try:
        input_date = datetime.strptime(user_input, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Please enter the date in the format YYYY-MM-DD.")
        continue

    # Generate dates for previous 10 days
    prev_dates = [input_date - timedelta(days=i) for i in range(1, look_back + 1)]

    # Filter the selected dataframe for these dates and features
    prev_data_df = df_selected[df_selected['datetime'].isin(prev_dates)].drop(columns=['datetime'])

    # Check if we have enough data
    if len(prev_data_df) < look_back:
        print("Insufficient historical data for the specified date.")
        continue

    # Scale and reshape the input data
    input_data_scaled = scaler.transform(prev_data_df)
    input_data_reshaped = np.reshape(input_data_scaled, (1, look_back, num_features))

    # Make predictions
    prediction = model.predict(input_data_reshaped)
    prediction_inversed = scaler.inverse_transform(prediction)

    # Display the prediction
    print(f"Weather prediction for {user_input}:")
    print(f"Cloud Cover: {prediction_inversed[0, 0]:.2f}")
    print(f"Maximum Temperature: {prediction_inversed[0, 1]:.2f}")
    print(f"Minimum Temperature: {prediction_inversed[0, 2]:.2f}")
    print(f"Precipitation: {prediction_inversed[0, 3]:.2f}")
    print(f"Sea Level Pressure: {prediction_inversed[0, 4]:.2f}")
    print(f"Snow Depth: {prediction_inversed[0, 5]:.2f}")
    # Add more fields if necessary
