import pandas as pd


insurance = pd.read_csv("https://raw.githubusercontent.com/EvidenceN/Insurance_premium_prediction/master/data/auto_insurance_data.csv")


pd.options.display.max_columns = 999


insurance.head()


insurance.shape


from pandas_profiling import ProfileReport


#profile = ProfileReport(insurance)


#profile


insurance.dtypes


insurance.head()


# Data Wrangling Function
# Change insurance effective to date column to datetime and store in new column called activation_date.
# Drop customer and Effective to date, # Response.

def Wrangle(dataframe):
    
    """Data Wrangling Function
    Change insurance effective to date 
    column to datetime and store in new column called 
    activation_date.
    Drop customer and Effective to date, # Response."""
    
    data = dataframe.copy()
    
    data["activation_date"] = pd.to_datetime(data["Effective To Date"], 
                                                  infer_datetime_format = True)
    
    data = data.drop(columns = ["Customer",
                                "Effective To Date",
                                'Response'])
    
    return data


insurance2 = Wrangle(insurance)


insurance2.head()


def split_data(dataframe):
    
    # split data into train, test, 
    # validation dataset before proceeding. 

    from sklearn.model_selection import train_test_split
    
    data = dataframe.copy()
    
    train, test = train_test_split(data, 
                                   train_size = 0.85, 
                                   test_size=0.15, 
                                   random_state=42)

    # validation dataset

    train, val = train_test_split(train, 
                                  train_size = 0.85, 
                                  test_size=0.15, 
                                  random_state=42)
    
    return train, test, val


train, test, val = split_data(insurance2)


val.head()


insurance2.shape


train.shape


test.shape


val.shape


def encode(train_data, test_data, val_data):
    
    # function to encode some columns using ordinal encoding, 
    # and other using onehotencoding. 
    
    train = train_data.copy()
    test = test_data.copy()
    val = val_data.copy()
    
    import category_encoders as ce
    
    # use ordinal encoding to do encode "coverage" column
    
    coverage_dictionary = [
        {'col': 'Coverage',
         'mapping':{"Basic":1, 
                    "Extended":2, 
                    "Premium": 3}}]

    coverage_encoder = ce.OrdinalEncoder(
        cols="Coverage", 
        mapping=coverage_dictionary)
    
    
    train_encoded = coverage_encoder.fit_transform(train)
    test_encoded = coverage_encoder.transform(test)
    val_encoded = coverage_encoder.transform(val)
    
    # combine college and bachelor into one datatype
    train_encoded['Education'] = train_encoded['Education'].replace(
        {"College":"Bachelor"})
    test_encoded['Education'] = test_encoded['Education'].replace(
        {"College":"Bachelor"})
    val_encoded['Education'] = val_encoded['Education'].replace(
        {"College":"Bachelor"})
    
    # encoding "education" column from categorical into integers

    education_dictionary = [
        {'col': 'Education',
         'mapping':{"High School or Below":1,
                    "Bachelor":2, 
                    "Master": 3,
                    "Doctor": 4}}]

    education_encoder = ce.OrdinalEncoder(
        cols="Education",
        mapping=education_dictionary)

    train_encoded = education_encoder.fit_transform(train_encoded)
    test_encoded = education_encoder.transform(test_encoded)
    val_encoded = education_encoder.transform(val_encoded)
    
    # encoding vehicle column from categorical into integers using ordinal encoding because
    # large, medium, small indicates a natural order. 

    vehicle_dictionary = [{'col': 'Vehicle Size',
                           'mapping':{"Small":1,
                                      "Medsize":2, 
                                      "Large": 3}}]

    vehicle_encoder = ce.OrdinalEncoder(cols="Vehicle Size", 
                                        mapping=vehicle_dictionary)

    train_encoded = vehicle_encoder.fit_transform(train_encoded)
    test_encoded = vehicle_encoder.transform(test_encoded)
    val_encoded = vehicle_encoder.transform(val_encoded)
    
    # one hot encoding columns
    columns_to_encode = ["Marital Status",
                         "Policy Type", 
                         "Policy", 
                         "Sales Channel", 
                         "Vehicle Class",
                         "Gender",
                         "Location Code",
                         "EmploymentStatus",
                         "Renew Offer Type",
                         "State"]
    
    
    # Encode columns above with one hot encoding
    # order doesn't matter in this columns situation

    columns_encoder = ce.OneHotEncoder(cols = columns_to_encode, 
                                       use_cat_names=True)

    train_encoded = columns_encoder.fit_transform(train_encoded)
    test_encoded = columns_encoder.transform(test_encoded)
    val_encoded = columns_encoder.transform(val_encoded)
    
    return train_encoded, test_encoded, val_encoded
    


train_encoded, test_encoded, val_encoded = encode(train, test, val)


train_encoded.head()


train.shape


train_encoded.shape


def change_name(train_encoded, test_encoded, val_encoded):
    
    """Renaming the columns after encoding into 
    something shorter. 
    
    The input is the encoded dataframe or any dataframe that
    needs to be renamed. """

    
    # creating a copy of the dataset. 
    train = train_encoded.copy()
    test = test_encoded.copy()
    val = val_encoded.copy()
    
    columns_to_rename = {
        "State_Arizona":"Arizona", 
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
         "Vehicle Class_Luxury Car": "Luxury Car",
         "Location Code_Urban": "Urban", 
         "Location Code_Rural": "Rural", 
         "Location Code_Suburban":"Suburban",
         "Gender_M": "Male", 
         "Gender_F": "Female",
         "EmploymentStatus_Employed": "Employed", 
         "EmploymentStatus_Unemployed": "Unemployed",
         "EmploymentStatus_Disabled": "Disabled", 
         "EmploymentStatus_Retired": "Retired",
         "EmploymentStatus_Medical Leave": "Medical_Leave",
         'Renew Offer Type_Offer2': 'Renew offer 1',
         'Renew Offer Type_Offer1': 'Renew offer 2',
         'Renew Offer Type_Offer3': 'Renew offer 3',
         'Renew Offer Type_Offer4': 'Renew offer 4'}
    
    train = train.rename(columns = columns_to_rename)
    test = test.rename(columns = columns_to_rename)
    val = val.rename(columns = columns_to_rename)
    
    return train, test, val


train2, test2, val2 = change_name(train_encoded, 
                                  test_encoded, 
                                  val_encoded)


train_encoded.head()


train2.head()


def round_values(train, test, val):
    
    """
    A function that will round the values in some columns
    
    provide the dataset, and the columns below will be rounded
    """
    
    # creating a copy of the dataset. 
    train_round = train.copy()
    test_round = test.copy()
    val_round = val.copy()
    
    train_round = train_round.round(
        {"Customer Lifetime Value": 2,
         "Total Claim Amount": 2})
    
    test_round = test_round.round(
        {"Customer Lifetime Value": 2,
         "Total Claim Amount": 2})
    
    val_round = val_round.round(
        {"Customer Lifetime Value": 2,
         "Total Claim Amount": 2})
    
    return train_round, test_round, val_round
    


train2, test2, val2 = round_values(train2, test2, val2)


train2.head()


## customer lifetime value and insurance premium as 
# target y variables. 


def x_y(train):
    
    """
    Function to split dataframe into x_values and y_values
    
    Define x variables and y variables"""
    
    y = train['Monthly Premium Auto']
    y2 = train['Customer Lifetime Value']
    
    # after initial models, these are the columns that has the 
    # most impact in predicting insurance premium and customer
    # lifetime value to insurance company
    
    important_columns = ["Luxury SUV", "Luxury Car", "Coverage", 
                    "Sports Car", "Two Door", "Four Door", 
                    "Months Since Policy Inception",
                    "Income", "Months Since Last Claim",
                    "Number of Policies", "Education", 
                    "Number of Open Complaints"]
    
    x = train[important_columns].copy()
    
    return y, y2, x
    


x_train.head()


# Training Data

y_train, y2_train, x_train = x_y(train2)


x_train.head()


# Testing Data

y_test, y2_test, x_test = x_y(test2)


# Validation Data

y_val, y2_val, x_val = x_y(val2)



