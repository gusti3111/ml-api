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
data["Fare"] = imputer.fit_transform(data[["Fare"]])



# Identify all categorical columns
categorical_columns = ["Sex", "Embarked", "Cabin", "Ticket", "Name"]  # Adjust this list based on your dataset


# Converting categorical data (example: Sex) to one-hot encoding
encoder = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
encoded_features = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns), index=data.index)



# Drop original categorical columns and concatenate encoded columns
data = data.drop(categorical_columns, axis=1)
data = pd.concat([data, encoded_df], axis=1)

# Scaling the 'Fare' feature
scaler = StandardScaler()
data[["Fare"]] = scaler.fit_transform(data[["Fare"]])

# Split data into train and test sets
X = data.drop("Survived", axis=1)
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choosing a LogisticRegression model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating Model Performance with accuracy, Precision, Recall, and F1-Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Save the model
with open("logistic_regression_model.pkl", "wb") as file:
    pickle.dump(model, file)

