import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import pickle


data = pd.read_csv("train.csv")


#handling missing values with mean

from sklearn.impute import SimpleImputer

imputer =SimpleImputer(strategy="mean")
data["Age"] = imputer.fit_transform(data[["Age"]])

#Converting categorical data (example: sex) to one-hot encoding
#One hot encoding is a technique that we use to represent categorical variables as numerical values in a machine learning model.

from sklearn.preprocessing import OneHotEncoder

encoder= OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[["Sex"]])
data = pd.concat([data, pd.DataFrame(encoded_features, columns=["Sex_encoded"])], axis=1)
data.drop("Sex", axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[["Fare"]] = scaler.fit_transform(data[["Fare"]])


