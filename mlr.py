import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv('50_Startups.csv')
    return dataset

# Main function
def main():
    # Set title
    st.title("Profit Prediction for Startups")

    # Load the dataset
    dataset = load_data()

    # Display the dataset
    st.subheader("Startup Dataset")
    st.write(dataset)

    # Separate features and target variable
    X = dataset.iloc[:, :-1]  # Features: R&D Spend, Administration, Marketing Spend, State
    y = dataset.iloc[:, -1]    # Target variable: Profit

    # Convert categorical variable 'State' to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    X['State'] = label_encoder.fit_transform(X['State'])

    # Encoding categorical variable 'State'
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X_encoded = ct.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.5, random_state=0)

    # Training the Multiple Linear Regression model on the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Checking the accuracy on the Test set
    accuracy = regressor.score(X_test, y_test)

    st.subheader("Model Evaluation")
    st.write("Accuracy:", accuracy)

    # Option to make predictions
    st.subheader("Make Predictions")
    st.write("Enter startup details to predict the profit:")

    rd_spend = st.number_input("R&D Spend")
    administration = st.number_input("Administration")
    marketing_spend = st.number_input("Marketing Spend")
    state = st.selectbox("State", dataset['State'].unique())

    # Convert the selected state to its corresponding label
    state_label = label_encoder.transform([state])[0]

    # Create a feature vector for prediction
    startup_details = np.array([[rd_spend, administration, marketing_spend, state_label]])

    # Encode the selected state
    startup_details_encoded = ct.transform(startup_details)

    # Make prediction
    if st.button("Predict"):
        prediction = regressor.predict(startup_details_encoded)
        st.write("Predicted Profit:", prediction[0])

if __name__ == "__main__":
    main()
