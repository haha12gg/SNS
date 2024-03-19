import pandas as pd
from datetime import datetime
# Load the files
file1_path = 'London2.csv'
file2_path = 'London3.csv'
file3_path = 'london_weather.csv'

# Read the files into DataFrames
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)
df3 = pd.read_csv(file3_path)

# Display the column names of each DataFrame to identify similar attributes
df1_columns = df1.columns.tolist()
df2_columns = df2.columns.tolist()
df3_columns = df3.columns.tolist()

# Rename columns in the third DataFrame to align with the first two
df3_renamed = df3.rename(columns={
    'date': 'datetime',
    'cloud_cover': 'cloudcover',
    'max_temp': 'tempmax',
    'mean_temp': 'temp',
    'min_temp': 'tempmin',
    'precipitation': 'precip',
    'pressure': 'sealevelpressure',
    'snow_depth': 'snowdepth'
})

# Convert 'datetime' in the third DataFrame from int to the format 'YYYY-MM-DD' (string)
df3_renamed['datetime'] = pd.to_datetime(df3_renamed['datetime'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

updated_file_path = 'London1.csv'  # Specify your own path
df3_renamed.to_csv(updated_file_path, index=False)

file_paths = {
    'London2':'London2_with_sunshine.csv',
    'London3':'London3_with_sunshine.csv'
}

# Make sea level pressure in same unit
for name, path in file_paths.items():
    df = pd.read_csv(path)

    # 调整sealevelpressure列，将小数点向右移两位
    df['sealevelpressure'] = df['sealevelpressure'] * 100

    # 输出新的CSV文件
    new_path = f'modified_{name}.csv'
    df.to_csv(new_path, index=False)
    print(f'Processed file saved to: {new_path}')

df = pd.read_csv('London3.csv')
df2 = pd.read_csv('London2.csv')
def calc_sunshine(row):
    sunrise = datetime.strptime(row['sunrise'], "%Y-%m-%dT%H:%M:%S")
    sunset = datetime.strptime(row['sunset'], "%Y-%m-%dT%H:%M:%S")
    sunshine_hours = (sunset - sunrise).total_seconds() / 3600
    return round(sunshine_hours, 1)

df['sunshine'] = df.apply(calc_sunshine, axis=1)
df2['sunshine'] = df2.apply(calc_sunshine, axis=1)
df.to_csv('London3_with_sunshine.csv', index=False)
df2.to_csv('London2_with_sunshine.csv', index=False)