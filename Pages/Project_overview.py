import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Project Overview",
    page_icon=":bar_chart:",
)

st.sidebar.image("andito_l_50371.jpg", width=100)  # Replace with your logo URL\
st.title("Project Overview")

st.subheader("Insurance Claims Data Analysis")
st.write("This project involves analyzing insurance claims data to identify patterns and insights that can help improve the claims process and reduce fraud. The analysis includes data cleaning, exploratory data analysis, and visualization of key metrics.")


st.write("## Insurance Claims Data")
df = pd.read_csv("C:/Users/gito2/Downloads/streamlit-demo-app-main/insurance_claims.csv")  # Assuming you have a CSV file with your projects
df.drop(["incident_hour_of_the_day",'insured_zip','policy_bind_date','incident_location'],axis=1,inplace=True)
st.write(df.head())  # Display the first few rows of the dataset

st.write("### Data Overview")
st.write("The dataset contains various features related to insurance claims, including policy details, incident information, and whether the claim was fraudulent or not. The goal is to analyze this data to uncover insights and trends.")
##Separating features and target variable
X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]
st.write("### Features and Target Variable")
st.write("Features (X):", X.columns.tolist())