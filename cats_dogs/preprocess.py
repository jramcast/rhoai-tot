import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# Identify Categorical and Numerical Columns
categorical_features = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


def preprocess(data: pd.DataFrame):
    # Step 2 - Data Pre-Processing
    X = data.drop(columns=["Target"])
    y = data["Target"]

    # Label Encode the Target Variables
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Ensure all Column Names are Strings
    X.columns = X.columns.astype(str)

    numerical_features = X.columns.difference(categorical_features).tolist()

    # Perform One-Hot Encoding to the Categorical Features
    onehot_encoder = OneHotEncoder(handle_unknown="ignore")
    x_encoded = onehot_encoder.fit_transform(X[categorical_features])

    # Create DataFrames from the Encoded Features
    x_encoded_df = pd.DataFrame(x_encoded, index=X.index)

    # Drop Original Categorical Columns and Concatenate the Encoded Columns
    X = X.drop(columns=categorical_features).reset_index(drop=True)
    X = pd.concat([X, x_encoded_df], axis=1)

    # Standardize Numerical Features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Splitting Training Dataset (80% Training Data and 20% Validation Data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83
    )
    return X_train, X_test, y_train, y_test
