import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error

# Load and combine datasets
file_paths = [
    'modified_London3.csv',
    'modified_London2.csv',
    'London13.csv'
]
dfs = [pd.read_csv(file_path) for file_path in file_paths]
combined_df = pd.concat(dfs, ignore_index=True)

# Data cleaning
combined_df.drop_duplicates(subset=['datetime'], inplace=True)
combined_df.sort_values('datetime', inplace=True)
combined_df = combined_df.interpolate(method='linear')

# Detect and remove outliers
z_scores = (combined_df - combined_df.mean()) / combined_df.std()
outliers = z_scores.abs() > 3
combined_df = combined_df[~outliers.any(axis=1)]

# Convert datetime to "MM-DD" format and add it as a feature
combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
combined_df['month_day'] = combined_df['datetime'].dt.strftime('%m-%d')

# Feature selection and engineering
features = ['month_day', 'cloudcover', 'tempmax', 'tempmin', 'precip', 'sealevelpressure', 'snowdepth', 'solarradiation']
df_selected = combined_df[features]

# One-hot encoding for month_day
df_selected = pd.get_dummies(df_selected, columns=['month_day'])

# Apply low-pass filter
values = df_selected.values

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# Prepare sequence data
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i])
    return np.array(X), np.array(y)

look_back = 4
X, y = create_sequences(scaled_data, look_back)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model architecture
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(look_back, X.shape[2])),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(y.shape[1])
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=12, batch_size=128, validation_data=(X_test, y_test), verbose=2)

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)

# RMSE calculation
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
print(f'Test RMSE: {rmse}')

# Save model and scaler
model.save("weather_prediction_model.h5")
joblib.dump(scaler, 'scaler.gz')

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
