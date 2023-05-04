#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initial imports.
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,balanced_accuracy_score,r2_score,mean_squared_error


# In[2]:


global_data_df = pd.read_csv('global_health_data.csv')
global_data_df["both_sex_life_exp"]=round(global_data_df["both_sex_life_exp"],3) # change this back to 3 decimal places
global_data_df


# In[3]:


mean=global_data_df["both_sex_life_exp"].mean()
stdev=global_data_df["both_sex_life_exp"].std()
print(mean)
print(stdev)
print(f"Use {round(mean+stdev,1)} years as the threshold for the high_life_exp group")
print(f"Use {round(mean-stdev,1)} years as the threshold for the low_life_exp group")


# In[4]:


# create a new column called life_expenctancy_group
bins = [0, 64.3, 79.4, 100]
labels = ['low', 'medium', 'high']
global_data_df['life_exp_group'] = pd.cut(global_data_df['both_sex_life_exp'], bins=bins, labels=labels)
global_data_df['life_exp_encoded'] = pd.cut(global_data_df['both_sex_life_exp'], bins=bins, labels=[0, 1, 2], include_lowest=True)
global_data_df


# In[5]:


# drop the country_code and country columns from the data set
global_data=global_data_df.drop(columns=["country_code","country","fem_life_exp","male_life_exp", "o3"])


# In[6]:


global_data = global_data.dropna()
global_data


# In[7]:


X = global_data.drop(columns=["both_sex_life_exp","life_exp_group","life_exp_encoded"])
X


# In[8]:


y = global_data["life_exp_encoded"]
y_1 = global_data["both_sex_life_exp"]


# In[9]:


# Splitting into Train and Test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)


# In[10]:


# Try to do StandardScaler later in this same cell.
# Creating a StandardScaler instance.
scaler = StandardScaler() # use this for StandardScaler scaling
#minmaxscaler = MinMaxScaler()
# Fitting the Standard Scaler with the training data.
X_scaler = scaler.fit(X_train) # standardscaler
# X_minmaxscaler = minmaxscaler.fit(X_train)
# Scaling the data.
## For StandardScaler scaling, sub X_minmaxscaler for X_scaler
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# In[11]:


# Instantiate a LogisticRegression model
model = LogisticRegression(solver = 'lbfgs',random_state=1)


# In[12]:


# Fitting the model
#rf_model = rf_model.fit(X_train_scaled, y_train)
model = model.fit(X_train_scaled, y_train)


# In[13]:


# Making predictions using the testing data.
y_pred = model.predict(X_test_scaled)


# In[14]:


# Print the accuracy score for the training and testing data for the LogisticRegression ML model
print(model.score(X_train_scaled, y_train))
print(model.score(X_test_scaled, y_test))


# In[15]:


from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)


# In[16]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[17]:


# Classification report
print(classification_report(y_test, y_pred))


# In[18]:


# Categorical predictions
y_test_df = pd.DataFrame(y_test)
y_test_df["predicted_values"]=y_pred
y_test_df=y_test_df.sort_index()


# In[19]:


# isolate countries with predictions:
predicted_countries = global_data_df[["country_code","country","life_exp_group"]]


# In[20]:


# merge the predicted_df with the original dataset using the index values
categorical_life_exp_predictions = predicted_countries.join(y_test_df, how='inner')
categorical_life_exp_predictions


# In[21]:


"""best_r2_score = 0
for i in range(50,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y_1, random_state=99)
    rf_model = RandomForestRegressor(n_estimators = 1024, random_state = 99)
    rf_model = rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    if r2 > best_r2_score:
        best_r2_score = r2
        best_condition = i
print(f"The best r2 score is {best_r2_score} and comes from random_state={i}")
"""


# In[22]:


# Predicting Continous Data
# Splitting into Train and Test sets. Specifically uses y_1 as the original values for life expectancy (continuous)
X_train, X_test, y_train, y_test = train_test_split(X, y_1, random_state=78)


# In[23]:


# Try to do StandardScaler later in this same cell.
# Creating a StandardScaler instance.
scaler = StandardScaler() # use this for StandardScaler scaling
#minmaxscaler = MinMaxScaler()
# Fitting the Standard Scaler with the training data.
X_scaler = scaler.fit(X_train) # standardscaler
# X_minmaxscaler = minmaxscaler.fit(X_train)
# Scaling the data.
## For StandardScaler scaling, sub X_minmaxscaler for X_scaler
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# In[24]:


# Create a random forest regressor. Change lasso back to RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators = 100, random_state = 78)
rf_model = rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)


# In[25]:


# Print the accuracy score for training and testing data for the RandomForestRegressor
print(rf_model.score(X_train_scaled, y_train))
print(rf_model.score(X_test_scaled, y_test))


# In[26]:


mse = round(mean_squared_error(y_test,y_pred),2)
print(f"The mean squared error is: {mse}") 


# In[27]:


# Test the accuracy of the RandomForestRegressor model with R^2
r2 = r2_score(y_test, y_pred)
r2


# In[28]:


# Continuous Life expectancy (floats) predictions
y_test_df = pd.DataFrame(y_test)
y_test_df["predicted_values"]=y_pred
y_test_df=y_test_df.sort_index()


# In[29]:


# merge the predicted_df with the original dataset using the index values
continuous_life_exp_predictions = predicted_countries.join(y_test_df, how='inner')
continuous_life_exp_predictions


# In[30]:


# Next step is look for correlations in the data
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import scipy
from sklearn.linear_model import LinearRegression
linearregression_model = LinearRegression()
linearregression_model.fit(X_train_scaled, y_train)

# Use our model to predict a value
predicted = linearregression_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)

print(f"mean squared error (MSE): {mse}")
print(f"R-squared (R2 ): {r2}")



# In[31]:


coeffs = linearregression_model.coef_
coeffs
corrs = np.corrcoef(X_train.T, y_train)
pearson_coeffs = corrs[:-1, -1]
pearson_coeffs


# In[32]:


coeffs = [x for x in pearson_coeffs]
correlations_data = global_data.drop(columns=["both_sex_life_exp","life_exp_group","life_exp_encoded"])
correlations_data
cols = [x for x in correlations_data.columns]
data = {'features': cols, 'r_values': coeffs}
correlations_df=pd.DataFrame(data)
correlations_df


# In[33]:


for i,row in correlations_df.iterrows():
    feature=row["features"]
    r = row["r_values"]
    n=38
    df=n-2
    t = r * math.sqrt(df / (1 - r**2))
    p_value = 1 - scipy.stats.t.cdf(t, df)
    #row["p_value"]=p_value
    correlations_df.at[i, "p_value"] = p_value
    print(f"The p-value for {feature} feature correlation with life expectancy is: {p_value}")


# In[34]:


correlations_df=correlations_df.sort_values("p_value",ascending=True)
print(correlations_df)

