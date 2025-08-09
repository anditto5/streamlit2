import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt 
from utils import load_css

load_css()

st_setup = st.set_page_config(
    page_title="Data Analysis",
    page_icon=":bar_chart:",
    layout="wide",
)

st.sidebar.image("andito_l_50371.jpg", width=100)  # Replace with your logo URL
st.title("Data Analysis")
st.subheader("Insurance Claims Data Analysis")
st.write("This project involves analyzing insurance claims data to identify patterns and insights that can help improve the claims process and reduce fraud. The analysis includes data cleaning, exploratory data analysis, and visualization of key metrics.")
df = pd.read_csv("C:/Users/gito2/Downloads/streamlit-demo-app-main/insurance_claims.csv")  # Assuming you have a CSV file with your projects
df.drop(["incident_hour_of_the_day", 'insured_zip', 'policy_bind_date', 'incident_location'], axis=1, inplace=True)



x= df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]

 # Display the count of fraud and non-fraud cases

cat_df = x.select_dtypes(include = ['object'])

st.write("Fraud Reported count")
fraud_count = y.value_counts()
st.write(fraud_count) 

#fraud_reported vs other features
st.write("#### Categorical Features")
st.write("The dataset contains various features related to insurance claims, including policy details, incident information, and whether the claim was fraudulent or not. The goal is to analyze this data to uncover insights and trends.")

for column in cat_df.columns:
    st.write(f"**{column}**")
    st.bar_chart(cat_df[column].value_counts())
st.write("#### Numerical Features")
num_df = x.select_dtypes(include=[np.number])
X = pd.concat([num_df, cat_df], axis = 1)
st.write(X.head())  # Display the first few rows of the combined dataset

st.write("### Fraud Reported vs Other Features")
st.write("The following visualizations provide insights into the distribution of numerical features in the dataset. These plots help identify patterns and relationships between features and the target variable (fraud_reported).")

# Visualizing the relationship between 'fraud_reported' and other features
# Calculate the number of columns to plot (all except 'fraud_reported')
# Calculate layout
n_cols_to_plot = len(df.columns) - 1
n_rows = (n_cols_to_plot + 4) // 5

fig, axes = plt.subplots(nrows=n_rows, ncols=5, figsize=(25, n_rows * 5))
axes = axes.flatten()

plot_index = 0
for col in df.columns:
    if col != 'fraud_reported':
        ax = axes[plot_index]
        if df[col].dtype == 'object':
            sns.countplot(x=col, hue='fraud_reported', data=df, ax=ax)
            ax.tick_params(axis='x', rotation=90)
        else:
            sns.histplot(data=df, x=col, hue='fraud_reported', kde=True, ax=ax)
        ax.set_title(f'{col} vs Fraud Reported')
        plot_index += 1

# Hide unused subplots
for j in range(plot_index, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# ✅ Correct way to display in Streamlit
st.pyplot(fig)