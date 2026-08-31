import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#used to fill in missing values
from sklearn.impute import SimpleImputer
#used to convert categorical.text asnwers into numeric columns
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/responses.csv")

#temp target
target = "Rock"

#define target and features used for prediction
y = df[target]
X = df.drop(columns=[target])

#split the traning and testing groups 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state =42
)

# Remove people whose target answer is missing.
train_valid = y_train.notna()
test_valid = y_test.notna()

X_train = X_train[train_valid]
y_train = y_train[train_valid]

X_test = X_test[test_valid]
y_test = y_test[test_valid]

#seperate numeric columns from text/categorical columns
numeric_columns = X_train.select_dtypes(include="number").columns
categorical_columns = X_train.select_dtypes(exclude='number').columns

# print("\nNumeric columns:", len(numeric_columns))
# print("Categorical columns:", len(categorical_columns))

#create imputer that replaces missing numeric values with median from training data
numeric_imputer = SimpleImputer(strategy='median')

#get median of each numerical column using only training set
numeric_imputer.fit(X_train[numeric_columns])

#use learned medians to fill missing numeric values 
X_train_numeric = numeric_imputer.transform(X_train[numeric_columns])
X_test_numeric = numeric_imputer.transform(X_test[numeric_columns])

#create imputer for categorical columns
#missing categorical asnwers will be filled in with the most common answer in the training data
categorical_imputer = SimpleImputer(strategy="most_frequent")

categorical_imputer.fit(X_train[categorical_columns])

X_train_categorical = categorical_imputer.transform(
    X_train[categorical_columns]
)

X_test_categorical = categorical_imputer.transform(X_test[categorical_columns])


#convert categorical snwers into numerical 0/1 columns
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

#learn which categories exist using only the training data
encoder.fit(X_train_categorical)

#convert categorical training and testing data into numbers
X_train_categorical_encoded = encoder.transform(X_train_categorical)
X_test_categorical_encoded = encoder.transform(X_test_categorical)

#combine cleaned numeric features and encoded categorical features into one complete input matrix for ML model
X_train_processed = np.hstack([
    X_train_numeric,
    X_train_categorical_encoded
])

X_test_processed = np.hstack([
    X_test_numeric,
    X_test_categorical_encoded
])