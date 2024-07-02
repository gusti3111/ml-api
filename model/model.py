import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# Load data
data = pd.read_csv("train.csv")

# Handling missing values with mean
imputer = SimpleImputer(strategy="mean")
data["Age"] = imputer.fit_transform(data[["Age"]])

# Converting categorical data (example: sex) to one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[["Sex"]])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(["Sex"]), index=data.index)
data = pd.concat([data, encoded_df], axis=1)
data.drop("Sex", axis=1, inplace=True)

# Scaling the 'Fare' feature
scaler = StandardScaler()
data[["Fare"]] = scaler.fit_transform(data[["Fare"]])


#split data train and test
X =data.drop("Survived",axis=1)
y= data["Survived"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(y_train)

# Check the result

