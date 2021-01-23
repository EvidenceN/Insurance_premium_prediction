import pandas as pd


insurance = pd.read_csv("https://raw.githubusercontent.com/EvidenceN/Insurance_premium_prediction/master/data/auto_insurance_data.csv")


pd.options.display.max_columns = 999


insurance.head()


insurance.shape


# from pandas_profiling import ProfileReport


#profile = ProfileReport(insurance)


#profile


insurance.dtypes


# Convert Effective_To_Date from categorical to date
# Effective_to_date could mean the day insurance starts or could be interpreted
# as the day insurance ends. As in effective until this date. 

insurance["activation_date"] = pd.to_datetime(insurance["Effective To Date"], infer_datetime_format = True)


insurance.dtypes


insurance.head()


insurance = insurance.drop(columns = ["Customer", "Effective To Date"])


insurance.head()


# split data into train, test, validation dataset before proceeding. 
# should have done this before chaging to datetime and dropping columns. 

from sklearn.model_selection import train_test_split

train, test = train_test_split(insurance, train_size = 0.85, test_size=0.15, random_state=42)

# validation dataset

train, val = train_test_split(train, train_size = 0.85, test_size=0.15, random_state=42)


train.shape


test.shape


val.shape


#encode coverage column from categorical to integer

insurance['Coverage'].describe()


insurance['Coverage'].value_counts()


#[{‘col’: ‘col1’, ‘mapping’: {None: 0, ‘a’: 1, ‘b’: 2}}] correct mapping structure for ordinal encoding

coverage_dictionary = [{'col': 'Coverage','mapping':{"Basic":1, "Extended":2, "Premium": 3}}]


# use ordinal encoding to do encode coverage column

import category_encoders as ce

coverage_encoder = ce.OrdinalEncoder(cols="Coverage", mapping=coverage_dictionary)


train_encoded = coverage_encoder.fit_transform(train)
test_encoded = coverage_encoder.transform(test)


train_encoded.head()


train_encoded['Coverage'].describe()


train['Coverage'].value_counts()


train_encoded['Coverage'].value_counts()


test['Coverage'].value_counts()


test_encoded['Coverage'].value_counts()


# encode education from categorical value to integers

train["Education"].describe()


train["Education"].value_counts()


# combine college and bachelor into one datatype. Could be that
# college means people that went to college but didn't graduate
# Combine college and bachelor for clarification and easier assessment

train['Education'] = train['Education'].replace({"College":"Bachelor"})


train["Education"].value_counts()


test['Education'] = test['Education'].replace({"College":"Bachelor"});


test["Education"].value_counts()


# encoding education column from categorical into integers

education_dictionary = [{'col': 'Education','mapping':{"High School or Below":1, 
                                                       "Bachelor":2, "Master": 3,
                                                      "Doctor": 4}}]

education_encoder = ce.OrdinalEncoder(cols="Education", mapping=education_dictionary)


train_encoded['Education'] = train_encoded['Education'].replace({"College":"Bachelor"})
test_encoded['Education'] = test_encoded['Education'].replace({"College":"Bachelor"})

train_encoded = education_encoder.fit_transform(train_encoded)
test_encoded = education_encoder.transform(test_encoded)


train_encoded['Education'].value_counts()


train_encoded.columns


# rounding various series in our dataframe. 
# columns to round - lifetime value, monthly premium, total claim amount

train_encoded = train_encoded.round({"Customer Lifetime Value": 2, "Total Claim Amount": 2})
test_encoded = test_encoded.round({"Customer Lifetime Value": 2, "Total Claim Amount": 2})
train = train.round({"Customer Lifetime Value": 2, "Total Claim Amount": 2})
test = test.round({"Customer Lifetime Value": 2, "Total Claim Amount": 2})


train_encoded.head()


train_encoded["EmploymentStatus"].describe()


train_encoded["EmploymentStatus"].value_counts()


# Encode employment status column with one hot encoding
# experiment with target encoding later. 
# order doesn't matter in this column situation

employment_encoder = ce.OneHotEncoder(cols = "EmploymentStatus", use_cat_names=True)

train_encoded = employment_encoder.fit_transform(train_encoded)
test_encoded = employment_encoder.transform(test_encoded)


train_encoded.head()


test_encoded.head()


# changing employement status column names on train dataset

train_encoded = train_encoded.rename(columns = {"EmploymentStatus_Employed": "Employed", "EmploymentStatus_Unemployed": "Unemployed",
                                      "EmploymentStatus_Disabled": "Disabled", "EmploymentStatus_Retired": "Retired",
                                      "EmploymentStatus_Medical Leave": "Medical_Leave"})


train_encoded.head()


# changing employement status column names on test dataset

test_encoded = test_encoded.rename(columns = {"EmploymentStatus_Employed": "Employed", "EmploymentStatus_Unemployed": "Unemployed",
                                      "EmploymentStatus_Disabled": "Disabled", "EmploymentStatus_Retired": "Retired",
                                      "EmploymentStatus_Medical Leave": "Medical_Leave"})


# encode gender to be numerical. 

train['Gender'].value_counts()


# Encode gender column with one hot encoding
# order doesn't matter in this column situation

gender_encoder = ce.OneHotEncoder(cols = "Gender", use_cat_names=True)

train_encoded = gender_encoder.fit_transform(train_encoded)
test_encoded = gender_encoder.transform(test_encoded)


train_encoded.head()


# change gender column names on train and test dataset. 

train_encoded = train_encoded.rename(columns = {"Gender_M": "Male", "Gender_F": "Female"})
test_encoded = test_encoded.rename(columns = {"Gender_M": "Male", "Gender_F": "Female"})


train_encoded.head()


train["Location Code"].describe()


train["Location Code"].value_counts()


# Encode location code column with one hot encoding
# order doesn't matter in this column situation

location_encoder = ce.OneHotEncoder(cols = "Location Code", use_cat_names=True)

train_encoded = location_encoder.fit_transform(train_encoded)
test_encoded = location_encoder.transform(test_encoded)


train_encoded.head()


# change location code column names on train and test dataset. 

train_encoded = train_encoded.rename(columns = {"Location Code_Urban": "Urban", "Location Code_Rural": "Rural", 
                                                "Location Code_Suburban":"Suburban"})
test_encoded = test_encoded.rename(columns = {"Location Code_Urban": "Urban", "Location Code_Rural": "Rural", 
                                                "Location Code_Suburban":"Suburban"})


train_encoded.head()


# change to integer these columns 

# marital status, sales_channel, state, vehicle size, type of policy, vehicle class, 
# 


train['Vehicle Size'].value_counts()


train['Policy'].describe()


# policy could have an order, but that order could also introduce bias, so, 
# one hot encoding was used for policy. 

# one hot encoding columns
columns_to_encode = ["Marital Status", "Policy Type", "Policy", "Sales Channel", "Vehicle Class"]

# ordinal encoding columns
vehicle_size_encoding = "Vehicle Size"


# encoding vehicle column from categorical into integers using ordinal encoding because
# large, medium, small indicates a natural order. 

vehicle_dictionary = [{'col': 'Vehicle Size','mapping':{"Small":1, 
                                                       "Medsize":2, "Large": 3}}]

vehicle_encoder = ce.OrdinalEncoder(cols="Vehicle Size", mapping=vehicle_dictionary)

train_encoded = vehicle_encoder.fit_transform(train_encoded)
test_encoded = vehicle_encoder.transform(test_encoded)


train_encoded.head()


# encode the remainder of columns using one hot encoder. 

columns_to_encode = ["Marital Status", "Policy Type", "Policy", "Sales Channel", "Vehicle Class"]

# Encode columns above with one hot encoding
# order doesn't matter in this columns situation

columns_encoder = ce.OneHotEncoder(cols = columns_to_encode, use_cat_names=True)

train_encoded = columns_encoder.fit_transform(train_encoded)
test_encoded = columns_encoder.transform(test_encoded)



train_encoded.head()


# drop 


# encode state using One Hot Encoding encoding without specifying order. 
# come back and experiment with different encoding later. 

state_encoder = ce.OneHotEncoder(cols="State", use_cat_names=True)

train_encoded = state_encoder.fit_transform(train_encoded)
test_encoded = state_encoder.transform(test_encoded)



train['State'].describe()


train_encoded.head()


# drop response and Renew Offer Type columns for reasons listed above

train_encoded = train_encoded.drop(columns=["Response", "Renew Offer Type"])
test_encoded = test_encoded.drop(columns=["Response", "Renew Offer Type"])


train_encoded.head()


test_encoded.head()


# change column names to something easier to pronounce

train_encoded = train_encoded.rename(columns = {"State_Arizona":"Arizona", 
                                     "State_Oregon": "Oregon",
                                     "State_California": "California",
                                     "State_Nevada": "Nevada",
                                     "State_Washington": "Washington", 
                                     "Marital Status_Married": "Married", 
                                     "Marital Status_Divorced":"Divorced",
                                     "Marital Status_Single": "Single",
                                     "Policy Type_Personal Auto": "Personal Auto",
                                     "Policy Type_Corporate Auto": "Corporate Auto",
                                     "Policy Type_Special Auto": "Special Auto",
                                     "Policy_Personal L3": "Personal L3", 
                                     "Policy_Corporate L3": "Corporate L3",
                                     "Policy_Corporate L2": "Corporate L2",
                                     "Policy_Corporate L1": "Corporate L1",
                                     "Policy_Personal L2": "Personal L2",
                                     "Policy_Personal L1": "Personal L1",
                                     "Policy_Special L1": "Special L1",
                                     "Policy_Special L2": "Special L2",
                                     "Policy_Special L3": "Special L3",
                                     "Sales Channel_Call Center": "Call Center",
                                     "Sales Channel_Agent": "Agent",
                                     "Sales Channel_Web": "Web",
                                     "Sales Channel_Branch": "Branch",
                                     "Vehicle Class_Two-Door Car": "Two Door",
                                     "Vehicle Class_SUV": "SUV",
                                     "Vehicle Class_Sports Car": "Sports Car",
                                     "Vehicle Class_Four-Door Car": "Four Door",
                                     "Vehicle Class_Luxury SUV": "Luxury SUV",
                                     "Vehicle Class_Luxury Car": "Luxury Car"})

test_encoded = test_encoded.rename(columns = {"State_Arizona":"Arizona", 
                                     "State_Oregon": "Oregon",
                                     "State_California": "California",
                                     "State_Nevada": "Nevada",
                                     "State_Washington": "Washington", 
                                     "Marital Status_Married": "Married", 
                                     "Marital Status_Divorced":"Divorced",
                                     "Marital Status_Single": "Single",
                                     "Policy Type_Personal Auto": "Personal Auto",
                                     "Policy Type_Corporate Auto": "Corporate Auto",
                                     "Policy Type_Special Auto": "Special Auto",
                                     "Policy_Personal L3": "Personal L3", 
                                     "Policy_Corporate L3": "Corporate L3",
                                     "Policy_Corporate L2": "Corporate L2",
                                     "Policy_Corporate L1": "Corporate L1",
                                     "Policy_Personal L2": "Personal L2",
                                     "Policy_Personal L1": "Personal L1",
                                     "Policy_Special L1": "Special L1",
                                     "Policy_Special L2": "Special L2",
                                     "Policy_Special L3": "Special L3",
                                     "Sales Channel_Call Center": "Call Center",
                                     "Sales Channel_Agent": "Agent",
                                     "Sales Channel_Web": "Web",
                                     "Sales Channel_Branch": "Branch",
                                     "Vehicle Class_Two-Door Car": "Two Door",
                                     "Vehicle Class_SUV": "SUV",
                                     "Vehicle Class_Sports Car": "Sports Car",
                                     "Vehicle Class_Four-Door Car": "Four Door",
                                     "Vehicle Class_Luxury SUV": "Luxury SUV",
                                     "Vehicle Class_Luxury Car": "Luxury Car"})


train_encoded.head()


# next week, re-factor ALL the data wrangling code and put everything in a GIANT function. 

# if there is still time, work on data visualization and exploration. 


train_encoded.head()


test_encoded.head()


# Explore relationship between month since last claim and insurance premium

# MONTH since last claim = Number of months since you made your last claim
# Monthly premium auto = Insurance premium

import plotly.express as px


# Explore relationship between month since last claim and insurance premium

# if month since last claim is 0, that means you recently had a claim
# if month since last claim is 35, that means you haven't had a clain in 35 months. 


px.scatter(train_encoded, 
            x = "Months Since Last Claim", 
            y = "Monthly Premium Auto", 
            trendline="ols")



# Look at relationship between month since last claim, and lifetime value

# if month since last claim is 0, that means you recently had a claim
# if month since last claim is 35, that means you haven't had a clain in 35 months. 


px.scatter(train_encoded, 
            x = "Months Since Last Claim", 
            y = "Customer Lifetime Value", 
            trendline="ols")


# Look at relationship between month since last claim, and lifetime value
# 3d scatter plot looking at lifetime value, insurance premium, and month since last claim. 


# customer lifetime value and insurance premium

px.scatter(train_encoded, 
            x="Customer Lifetime Value", 
            y="Monthly Premium Auto", 
            trendline="ols",
            color="Monthly Premium Auto")


# a sample of the data. 
train_sample = train_encoded.sample(n=100, random_state=42)
train_sample.shape


px.scatter_3d(train_sample, 
            x="Months Since Last Claim",
            y="Customer Lifetime Value", 
            z="Monthly Premium Auto",
            color="Months Since Last Claim")


train_sample.columns.to_list()


# Look at relationship between month since policy inception, and premium price

px.scatter(train_sample, x="Months Since Policy Inception", y="Monthly Premium Auto")


# Look at relationship between month since policy inception, and customer lifetime value

px.scatter(train_sample, x="Months Since Policy Inception", y="Customer Lifetime Value")


import seaborn as sns


# using seaborn to see from a high level perspective what the relationship between multiple variables. 

# sns.pairplot(train_sample)


# Next week, look at seaborn pairplot result, then determine what visualizations to create to further explore the data, then start building the model starting with "Mean baseline" for both targets we are working with. 


"""a = sns.pairplot(train_sample, vars=['Customer Lifetime Value', 
                                "Coverage", 
                                "Education", 
                                "Income",
                                "Monthly Premium Auto", 
                                'Months Since Last Claim', 
                                "Months Since Policy Inception", 
                                "Number of Open Complaints",
                                "Number of Policies", 
                                "Total Claim Amount",
                                "Vehicle Size"])"""


# focusing on interesting visualizations

"""b = sns.pairplot(train_sample, vars=['Customer Lifetime Value',
                                "Income",
                                "Monthly Premium Auto", 
                                'Months Since Last Claim', 
                                "Months Since Policy Inception", 
                                "Total Claim Amount"])"""


# look at total claim amount and monthly premium auto

px.scatter(train_sample, x='Monthly Premium Auto', y='Total Claim Amount')





# train test split x and y

# 2 separate y variables, 

# y1 = Monthly Premium Auto
# y2 = Customer Lifetime Value
# remove leakage feature = Total Claim Amount

"""
Total claim amount could be considered data leakage because when you first get an insurance premium, 
the new insurance company won't know how much you are going to claim in total claim amount
But, once you make claims, you insurance premium is going to go up in accordance to your claim amount

The objective is to predict what you insurance premium is going to be without first knowing
how much you are going to claim in claim amount. How much you are going to claim is "future" information
which means it is leaking data into our present time of what insurance premium is and customer lifetime value

"""





insurance.shape


train_encoded.shape


test_encoded.shape


# defining y_variables

y_train = train_encoded['Monthly Premium Auto']
y_test = test_encoded['Monthly Premium Auto']


print(len(y_train))
y_train


print(len(y_test))
y_test


train_encoded.head()


# define x variables
# drop leakage column and y variables. 

# columns to drop from x dataset. 

drop_columns = ["Monthly Premium Auto", "Customer Lifetime Value", "Total Claim Amount"]

x_train = train_encoded.drop(columns = drop_columns)
x_train.head()


x_train.shape


drop_columns = ["Monthly Premium Auto", "Customer Lifetime Value", "Total Claim Amount"]

x_test = test_encoded.drop(columns = drop_columns)
print(x_test.shape)
x_test.head()


# baseline model. 

# for regression problems, the mean serves as a good baseline
# for classification problems, the mode serves as a good baseline



# mean baseline

mean = y_train.mean()
round(mean, 2)

# without using a model we predict that insurance premium would be $93.49


y_train.describe()


error = mean - y_train
error


# mean absolute error of our mean baseline

mean_absolute_error = error.abs().mean()
mean_absolute_error
print(f'By guessing, our insurance premium would be ${round(mean, 2)} \nand we would be off by ${round(mean_absolute_error, 2)}')


# FIGURE OUT HOW TO MANUALLY CALCULATE MEAN SQUARED ERROR

"""import math

mean_squared_error = math.sqrt(error.mean())
mean_squared_error"""


from sklearn.linear_model import LinearRegression

lr = LinearRegression()


x_train


# dropp activation date column because linear regression model didn't like it. 
x_train = x_train.drop(columns = 'activation_date')


x_train


# dropp activation date column because linear regression model didn't like it. 
x_test = x_test.drop(columns = 'activation_date')


x_test


# fitting linear regression model

lrModel = lr.fit(x_train, y_train)


# predicting y_values using test dataset. 

y_pred = lrModel.predict(x_test)


y_pred


# compare our model prediction to the actual values. 
y_testA = list(y_test)

print(y_pred[0]) # predicted value for first row
print(y_testA[0]) # actual value for first row


from sklearn.metrics import mean_squared_error, mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'Linear Regression mean absolute error {mae}')
print(f"Linear Regression mean squared error {mse}")


# look at what coefficients contributed to the score in our linear regression model. 

coeff = lrModel.coef_
coeff


intercept = lrModel.intercept_
intercept


y_pred.mean()


# first see what features are most important in the linear regressio model

# and then build out something where you can input different numbers and get a prediction. 

# when I build the flask app, I want the users to input their data and get a prediction. 


# which features has significant impact in prediction auto insurance premium.



x_train.columns


# plotting the coefficients from linear regression model.
columns = x_train.columns

# series between columns and coefficients

lrGraph = pd.Series(coeff, columns)


lrGraph


# Visualization for Linear Regression model.
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(10,10))

lrGraph.sort_values().plot.barh(color='red')

plt.title('Visualization for Linear Regression Model Coefficients')


## Random Forest Model


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)

# fitting random regression model

rfModel = rf.fit(x_train, y_train)

# predicting y_values using test dataset. 

y_pred_r = rfModel.predict(x_test)

mae_r = mean_absolute_error(y_test, y_pred_r)
mse_r = mean_squared_error(y_test, y_pred_r)
print(f'Random Forest Regression mean absolute error {mae_r}')
print(f"Random Forest Regression mean squared error {mse_r}")



# feature importances from random forest model

importances = rfModel.feature_importances_
importances


# interpreting random forest model. 

importances = rfModel.feature_importances_

# columns used in random forest model.
columns = x_train.columns

# series between columns and feature importances

rfGraph = pd.Series(importances, columns)


# Visualization for Random Regression model.

figure(figsize=(10,10))

rfGraph.sort_values().plot.barh(color='red')

plt.title('Visualization for Random Forest Regression Model Feature Importances')


x_train.head()


rfGraph.sort_values(ascending=False)[:10]


top_10_features = ['Luxury SUV', 'Luxury Car', 'Coverage', 
                   'SUV', 'Sports Car', 'Two Door', 'Four Door', 
                   'Months Since Policy Inception', 'Income', 
                   'Months Since Last Claim']

x_train2 = x_train[top_10_features]

x_test2 = x_test[top_10_features]


x_train2.head()


x_test2.head()



from sklearn.ensemble import RandomForestRegressor

rf2 = RandomForestRegressor(random_state=42)

# fitting random regression model

rfModel2 = rf.fit(x_train2, y_train)

# predicting y_values using test dataset. 

y_pred_r2 = rfModel2.predict(x_test2)

mae_r2 = mean_absolute_error(y_test, y_pred_r2)
mse_r2 = mean_squared_error(y_test, y_pred_r2)
print(f'Random Forest Regression mean absolute error {mae_r2}')
print(f"Random Forest Regression mean squared error {mse_r2}")



# let's hyper parameter tune "Number of Estimators"
from sklearn.model_selection import GridSearchCV

gsModel = RandomForestRegressor(random_state=42)

params = {'n_estimators': [50, 100, 200, 300, 400, 500]}

search = GridSearchCV(estimator = gsModel,
                   param_grid = params,
                   n_jobs = -1)


# fitting grid search cv. 

search_model = search.fit(x_train2, y_train)


# best scores from the model. 

search_model.best_params_


y_pred_search = search_model.predict(x_test2)


mae_search = mean_absolute_error(y_test, y_pred_search)
mse_search = mean_squared_error(y_test, y_pred_search)
print(f'Grid Search CV mean absolute error {mae_search}')
print(f"Grid Search CV mean squared error {mse_search}")


# let's hyper parameter tune "Number of Estimators"

from sklearn.model_selection import GridSearchCV

gsModel = RandomForestRegressor(random_state=42)

params = {'n_estimators': [500, 600, 700, 800, 900, 1000]}

search = GridSearchCV(estimator = gsModel,
                   param_grid = params,
                   n_jobs = -1)

# fitting grid search cv. 

search_model = search.fit(x_train2, y_train)

# best scores from the model. 

print(f'Best parameters: {search_model.best_params_}')

y_pred_search = search_model.predict(x_test2)

mae_search = mean_absolute_error(y_test, y_pred_search)
mse_search = mean_squared_error(y_test, y_pred_search)
print(f'Grid Search CV mean absolute error {mae_search}')
print(f"Grid Search CV mean squared error {mse_search}")


search_model


# let's hyper parameter tune "maximum depth of tree"

from sklearn.model_selection import GridSearchCV

gsModel = RandomForestRegressor(random_state=42,
                               n_estimators = 700)

params = {'max_depth': [5, 10, 15, 20, 25]}

search = GridSearchCV(estimator = gsModel,
                   param_grid = params,
                   n_jobs = -1)

# fitting grid search cv. 

search_model = search.fit(x_train2, y_train)

# best scores from the model. 

print(f'Best parameters: {search_model.best_params_}')

y_pred_search = search_model.predict(x_test2)

mae_search = mean_absolute_error(y_test, y_pred_search)
mse_search = mean_squared_error(y_test, y_pred_search)
print(f'Grid Search CV mean absolute error {mae_search}')
print(f"Grid Search CV mean squared error {mse_search}")


# let's hyper parameter tune "maximum features to consider for each split"

from sklearn.model_selection import GridSearchCV

gsModel = RandomForestRegressor(random_state=42,
                               n_estimators = 700,
                               max_depth = 15)

params = {'max_features': [2, 4, 6, 8, 10]}

search = GridSearchCV(estimator = gsModel,
                   param_grid = params,
                   n_jobs = -1)

# fitting grid search cv. 

search_model = search.fit(x_train2, y_train)

# best scores from the model. 

print(f'Best parameters: {search_model.best_params_}')

y_pred_search = search_model.predict(x_test2)

mae_search = mean_absolute_error(y_test, y_pred_search)
mse_search = mean_squared_error(y_test, y_pred_search)
print(f'Grid Search CV mean absolute error {mae_search}')
print(f"Grid Search CV mean squared error {mse_search}")


# let's hyper parameter tune "maximum depth of tree"

from sklearn.model_selection import GridSearchCV

gsModel = RandomForestRegressor(random_state=42,
                               n_estimators = 700)

params = {'max_depth': [15, 20, None]}

search = GridSearchCV(estimator = gsModel,
                   param_grid = params,
                   n_jobs = -1)

# fitting grid search cv. 

search_model = search.fit(x_train2, y_train)

# best scores from the model. 

print(f'Best parameters: {search_model.best_params_}')

y_pred_search = search_model.predict(x_test2)

mae_search = mean_absolute_error(y_test, y_pred_search)
mse_search = mean_squared_error(y_test, y_pred_search)
print(f'Grid Search CV mean absolute error {mae_search}')
print(f"Grid Search CV mean squared error {mse_search}")


model = RandomForestRegressor(random_state=42,
                               n_estimators = 700,
                               max_depth = 15,
                               max_features=6)

model.fit(x_train2, y_train)

y_pred_final = model.predict(x_test2)

mae_final = mean_absolute_error(y_test, y_pred_final)
mse_final = mean_squared_error(y_test, y_pred_final)
print(f'Final Random Forest Regressor mean absolute error {mae_final}')
print(f"Final Random Forest Regressor mean squared error {mse_final}")


list(y_pred_final[:5])


list(y_test)[:5]


from sklearn.ensemble import GradientBoostingRegressor


# fitting gradient boosting model on entire x train without extracting top 10 features

gbModel = GradientBoostingRegressor(random_state = 42)

gbModel.fit(x_train, y_train)

y_pred_gb= gbModel.predict(x_test)

mae_gb= mean_absolute_error(y_test, y_pred_gb)
mse_gb= mean_squared_error(y_test, y_pred_gb)
print(f'Gradient Boosting Regressor mean absolute error {mae_gb}')
print(f'Gradient Boosting Regressor mean squared error {mse_gb}')


## Interpreting Gradient boosting model. 

gbImportances = gbModel.feature_importances_

# columns used in gradient boosting model.
columns = x_train.columns

# series between columns and feature importances

gbGraph = pd.Series(gbImportances, columns)

# Visualization for gradient boosting Regression model.

figure(figsize=(10,10))

gbGraph.sort_values().plot.barh(color='red')

plt.title('Visualization for Gradient Boosting Regression Model Feature Importances')


# fitting gradient boosting model on top 10 features from x-train

gbModel2 = GradientBoostingRegressor(random_state = 42)

gbModel2.fit(x_train2, y_train)

y_pred_gb2= gbModel2.predict(x_test2)

mae_gb2= mean_absolute_error(y_test, y_pred_gb2)
mse_gb2= mean_squared_error(y_test, y_pred_gb2)
print(f'Gradient Boosting Regressor mean absolute error {mae_gb2}')
print(f'Gradient Boosting Regressor mean squared error {mse_gb2}')


x_train2.head()


# seting what coverage means. 
x_train2['Coverage'].describe()


# testing what prediction will be using dummy data. 


test_data = [[0, 0, 2, 0, 1, 0, 1, 20, 50000, 10]]


model.predict(test_data)


import shap


model = RandomForestRegressor(random_state=42,
                               n_estimators = 700,
                               max_depth = 15,
                               max_features=6)

model.fit(x_train2, y_train)

y_pred_final = model.predict(x_test2)

mae_final = mean_absolute_error(y_test, y_pred_final)
mse_final = mean_squared_error(y_test, y_pred_final)
print(f'Final Random Forest Regressor mean absolute error {mae_final}')
print(f"Final Random Forest Regressor mean squared error {mse_final}")


# function to do shapley plots. 

def shapley(x_train, y_train, x_test, row_number=0):
    
    model = RandomForestRegressor(random_state=42,
                               n_estimators = 700,
                               max_depth = 15,
                               max_features=6)
    
    model.fit(x_train, y_train)

    # defining what row to examine in the shapley plot
    row = x_test.iloc[[row_number]]

    # predicting
    pred = model.predict(row)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)

    feature_names = row.columns
    feature_values = row.values[0]
    shaps = pd.Series(shap_values[0], zip(feature_names, feature_values))

    # printing the result

    """result = f'{pred[0]:.1f} is the predicted Insurance Premium. \n\n'
    result += f'Starting from baseline of ${explainer.expected_value:.1f} \n\n'
    result += shaps.to_string()
    print(result)"""

    shap.initjs()

    return shap.force_plot(
      base_value = explainer.expected_value,
      shap_values = shap_values,
      features=row)




x_test2.head()


x_test2.iloc[0]


shapley(x_train2, y_train, x_test2)


x_test2.iloc[20]


shapley(x_train2, y_train, x_test2, row_number=20)


# define new y_target, and new x_target. 
# then do a baseline model
# then do a linear regression model
# linear regression model explanation
# then do a random forest model
# random forest model explanation. 


# define x and y values for customer lifetime value

# keeping x_train and x_test the same. 

x_train_ltv = x_train
x_test_ltv = x_test

# defining y_variables

y_train_ltv = train_encoded['Customer Lifetime Value']
y_test_ltv = test_encoded['Customer Lifetime Value']


len(y_test_ltv)


# mean baseline = random guess. 
# If I were to guess what the lifetime value of the customer is, 
# this will be my guess. 

mean_ltv = y_train_ltv.mean()
round(mean_ltv, 2)


error_ltv = mean_ltv - y_train_ltv

# mean absolute error of our mean baseline

mean_ltv_absolute_error = error_ltv.abs().mean()
print(f'By guessing, our customer lifetime value would be ${round(mean_ltv, 2)} \nand we would be off by ${round(mean_ltv_absolute_error, 2)}')


# linear regression model for customer lifetime value

lr_ltv = LinearRegression()

# fitting linear regression model

lr_ltvModel = lr_ltv.fit(x_train_ltv, y_train_ltv)

# predicting y_values using test dataset. 

y_pred_ltv = lr_ltvModel.predict(x_test_ltv)

mae_ltv = mean_absolute_error(y_test_ltv, y_pred_ltv)
mse_ltv = mean_squared_error(y_test_ltv, y_pred_ltv)
print(f'Linear Regression mean absolute error {mae_ltv}')
print(f"Linear Regression mean squared error {mse_ltv}")


# visualizing coefficients for customer lifetime value using linar regression

coeff_ltv = lr_ltvModel.coef_

# plotting the coefficients from linear regression model.
columns = x_train_ltv.columns

# series between columns and coefficients

lr_ltvGraph = pd.Series(coeff_ltv, columns)

# Visualization for Linear Regression model.
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(10,10))

lr_ltvGraph.sort_values().plot.barh(color='red')

plt.title('Visualization for Linear Regression Model Coefficients')


# build a random forest model for customer lifetime value. 

from sklearn.ensemble import RandomForestRegressor

rf_ltv = RandomForestRegressor(random_state=42)

# fitting random regression model

rf_ltvModel = rf_ltv.fit(x_train_ltv, y_train_ltv)

# predicting y_values using test dataset. 

y_pred_r_ltv = rf_ltvModel.predict(x_test_ltv)

mae_r_ltv = mean_absolute_error(y_test_ltv, y_pred_r_ltv)
mse_r_ltv = mean_squared_error(y_test_ltv, y_pred_r_ltv)
print(f'Random Forest Regression mean absolute error {mae_r_ltv}')
print(f"Random Forest Regression mean squared error {mse_r_ltv}")


# visualize which features contribed to random forest prediction. 

# interpreting random forest model. 

importances_ltv = rf_ltvModel.feature_importances_

# columns used in random forest model.
columns = x_train.columns

# series between columns and feature importances

rf_ltvGraph = pd.Series(importances_ltv, columns)

# Visualization for Random Regression model.

figure(figsize=(10,10))

rf_ltvGraph.sort_values().plot.barh(color='red')

plt.title('Visualization for Random Forest Regression Model Feature Importances')


shapley(x_train_ltv, y_train_ltv, x_test_ltv)
