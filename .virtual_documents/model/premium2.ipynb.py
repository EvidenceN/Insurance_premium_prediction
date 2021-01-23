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
         "EmploymentStatus_Medical Leave": "Medical_Leave"}
    
    train = train.rename(columns = columns_to_rename)
    test = test.rename(columns = columns_to_rename)
    val = val.rename(columns = columns_to_rename)
    
    return train, test, val


train2, test2, val2 = change_name(train_encoded, 
                                  test_encoded, 
                                  val_encoded)


train_encoded.head()


train2.head()


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


# change gender column names on train and test dataset. 

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
