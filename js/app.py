from flask import Flask, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

app = Flask(__name__)



# Load the dataset
global_data_df = pd.read_csv('global_health_data.csv')

# Preprocess the data
global_data_df["both_sex_life_exp"] = round(global_data_df["both_sex_life_exp"], 3)
bins = [0, 64.3, 79.4, 100]
labels = ['low', 'medium', 'high']
global_data_df['life_exp_group'] = pd.cut(global_data_df['both_sex_life_exp'], bins=bins, labels=labels)
global_data_df['life_exp_encoded'] = pd.cut(global_data_df['both_sex_life_exp'], bins=bins, labels=[0, 1, 2], include_lowest=True)
global_data = global_data_df.drop(columns=["country_code","country","fem_life_exp","male_life_exp", "o3"])
global_data = global_data.dropna()

# Split the data into features and target
X = global_data.drop(columns=["both_sex_life_exp","life_exp_group","life_exp_encoded"])
y_1 = global_data["both_sex_life_exp"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_1, random_state=78)

# Scale the data
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    co = float(request.form.get('co'))
    no = float(request.form.get('no'))
    no2 = float(request.form.get('no2'))
    so2 = float(request.form.get('so2'))
    pm2_5 = float(request.form.get('pm2_5'))
    pm10 = float(request.form.get('pm10'))
    gdp_per_capita = float(request.form.get('gdp_per_capita'))
    mean_bmi = float(request.form.get('Mean BMI'))
    che_per_capita_usd = float(request.form.get('che_per_capita_usd'))
    poverty_rate = float(request.form.get('poverty_rate'))
    country = request.form.get('country')
    
    # Find the corresponding row in the dataset based on the input country
    row_idx = global_data[global_data['country'] == country].index[0]
    
    # Make a copy of the dataset and update the values
    new_data = global_data.copy()
    new_data.loc[row_idx, 'co'] += co
    new_data.loc[row_idx, 'no'] += no
    new_data.loc[row_idx, 'no2'] += no2
    new_data.loc[row_idx, 'so2'] += so2
    new_data.loc[row_idx, 'pm2_5'] += pm2_5
    new_data.loc[row_idx, 'pm10'] += pm10
    new_data.loc[row_idx, 'gdp_per_capita'] += gdp_per_capita
    new_data.loc[row_idx, 'mean_bmi'] += mean_bmi
    new_data.loc[row_idx, 'che_per_capita_usd'] += che_per_capita_usd
    new_data.loc[row_idx, 'poverty_rate'] += poverty_rate

    # Split the data into features and target
    X_new = new_data.drop(columns=["both_sex_life_exp","life_exp_group","life_exp_encoded"])
    y_new = new_data["both_sex_life_exp"]

    # Scale the new data
    X_new_scaled = X_scaler.transform(X_new)

    # Make a prediction based on the new data and the original linear regression model
    new_pred = linear_model.predict(X_new_scaled)[0]
    old_pred = linear_model.predict(X_test_scaled)[0]

    # Calculate the change in life expectancy
    change = new_pred - old_pred

    # Return the predicted change in life expectancy to the user
    return f"{change:.3f}"