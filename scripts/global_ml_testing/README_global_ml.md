## Global Data Machine Learning
I made categorical and continuous predictions each for country's life expectancy values.

### Categorical ML Model Predictions
For the categorical predictions, I created three life expectancy groups: low, medium, and high.
I took the mean value of the life expectancies from all countries. The global life expectancy (mean) value was 71.9 years. I then designated the low life expectancy group to consist of life expectancy values less than one standard deviation below the global life expectancy mean value. 1 SD below the mean was 64.3 years I designated the high life expectancy group to consist of life expectancy values greater than one standard deviation above the mean at a value of 79.4 years. The medium life expectancy group consisted of values in the range of +/- 1 SD from the mean.
Then, I manually encoded the life_expectancy groups by setting the "low" group equal to number 0, the "medium" group equal to number 1, and the "high" group equal to number 2. Now, the life expectancy values were properly encoded for cateogrical data predictions with machine learning models.
Only numerical values were selected from the parental dataset and stored in the variable `X`. The value for life_exp_encoded was set equal to the variable `y`.

The dataset X,y was then split up into training and testing data using `train_test_split` from sklearn.
The data was X_train and X_test data was transformed using StandardScaler.
The LogisticRegression ML model was used for making categorical predictions for each country's life expectancy group.
The resulting accuracy scores for LogisticRegression on the training data was 0.86 and testing data was 0.87.
This produced a balanced_accuracy_score value of about 0.94.

### Continuous Data (Floating Point Values for Life Expectancy) ML Model Predictions
For the continuous predictions, I maintained the original floating point number's for each countries life expectancy.
I used the variable `X` for these predictions. A new variable `y_1` was set equal to the original floating point numbers for country life expectancy.
The data was split using `train_test_split` from sklearn.
The data was transformed using StandardScaler.
The RandomForestRegressor model was used to make continuous data predictions for life expectancy.
The resulting accuracy scores for the training and testing data were 0.97 and 0.83, respectively. This meant that there was slight overfitting of the model on the training data.
Since this function relies on an advanced linear regression, an r2_score can also accurately assess the performance of the RandomForestRegressor model.
The resulting r2_score was 0.83. This is considered a good r^2 value. Thus, the RandomForestRegressor model fit the data pretty well and made accurate predictions with the testing data.

### Multiple Linear Regression Model to Observe Correlations Between Feature and Target Data.
The LinearRegression model from sklearn library was used to determine correlations between feature data and life expectancy values.
The same split, and scaled data from the RandomForestRegressor model was used (X and y_1)
The model was fit to X_train_scaled and y_train. the LinearRegression model made predictions `predicted` using the linearregression_model.predict() function.
A resulting r^2 value of 0.63 was obtained with a mean squared error of 17.78.
The `coef_` function was applied to the LinearRegression model in order to observe the slope coefficient values for feature data correlation with life_expectancy data.
The pearson coefficient values were obtained from the initial coefficient values using th  np.corrcoef() function.
This function computes the Pearson correlation coefficient matrix of the features. The last value in each row of the resulting corrs variable is the pearson coefficient (r-value).
The r-values for each variable in the feature data were compiled into the final table.
Statistically significant r-values were observed for gdp_per_capita, obesity, and no2.
Logically, a country with a high gdp_per_capita value can likely predict a longer life expectancy.
Interestingly, the obesity parameter also observed a positive and statistically significant correlation with longer life expectancy.
This could be explained by the fact that a more prosperous and successful nation has plenty of food to eat. This could lead to an increased average BMI index value for a given country. However, BMI might not be the best indicator for obesity. BMI only factors in body weight divided by height. This does not include body fat percent or muscle density.
Also, no2 had a statistically significant correlation with prolonged life expectancy. Although the correlation is weak at a pearson coefficient (r-value) of around 0.36.
It could be that more prosperous nations have the ability to build more power plants and burn more fossil fuels. And thus as a result, the general technological advances of a nation, despite its pollution byproducts, help contribute to a longer life expectancy.
