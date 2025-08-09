import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder



st_setup = st.set_page_config(
    page_title="Profile",
    page_icon=":bar_chart:",
    layout="wide",
)

st.sidebar.image("andito_l_50371.jpg", width=100)  # Replace with your logo URL)


st.title("Profile : Anditto Farhan Waskito Putra")
st.subheader("Data Scientist | Machine Learning Engineer | Python Developer")
st.write("This is my personal profile page where you can learn more about my skills, experience, and projects. I am passionate about data science and machine learning, and I enjoy working on challenging problems that require innovative solutions.")
st.write("Feel free to explore and connect with me on (www.linkedin.com/in/anditto-farhan-waskitoputra).")
st.write("E-mail me at: andittofa@gmail.com")
st.write("Check out my GitHub for more projects:(https://github.com/anditto5)")
st.write("location: Indonesia, Kota Wisata Cibubur cileungsi Bogor, West Java")

st.write("## Professional summary")
st.write("""I am a passionate data scientist with a strong foundation in machine learning and Python development. I have experience in building  model for ECG Image Detection and insurance fraud. My expertise includes data analysis, model training, and optimization. I am always eager to learn new technologies and improve my skills in the field of data science.""")

st.write("## Skills")
st.write(" Hard skills = [Python, Machine Learning, Deep Learning, Data Analysis, Model Deployment, Data Visualization, SQL, Pandas, NumPy, Scikit-learn, TensorFlow, Keras, Streamlit]")
st.write("Soft skills = [Problem Solving, Critical Thinking, Communication, Teamwork, Adaptability, Time Management, Attention to Detail]")

st.write("## Languages")
st.write("English (Fluent), Bahasa Indonesia (Native), Japanese (basic)")


st.write("## Education")
st.write("Bachelor of Mechanical Engineering, Diponegoro University, 2014-2021, Semarang, Indonesia, GPA: 2.79/4.0")
st.write("Master of Data Science, Universitas Indonesia, 2022-2024, Depok, Indonesia, GPA: 3.19/4.0")

st.write("## Certifications")
st.write("•	Bootcamp Deep Learning for Medical Imaging" )
st.write("•	Rapid Data Science Mastery ")
st.write("•	Mastering python & EDA ( Unlocking Data Insight & with exploration data analysis )")

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


 # Display the count of fraud and non-fraud cases

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

num_df = x.select_dtypes(include=[np.number])
X = pd.concat([num_df, cat_df], axis = 1)

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

num_df = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss','number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]

# Scaling the numeric values in the dataset

scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)
scaled_num_df = pd.DataFrame(data = scaled_data, columns = num_df.columns, index = X_train.index)


X_train.drop(columns = scaled_num_df.columns, inplace = True)
X_train = pd.concat([X_train, scaled_num_df], axis = 1)




le = LabelEncoder()
y_train = le.fit_transform(y_train)
# Scale the numerical values in the test dataset
num_df_test = X_test[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss','number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]
num_df_train = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss','number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]

scaled_data_test = scaler.transform(num_df_test)
scaled_data_train = scaler.fit_transform(num_df_train)

scaled_num_df_test = pd.DataFrame(data = scaled_data_test, columns = num_df_test.columns, index = X_test.index)
scaled_num_df_train = pd.DataFrame(data=scaled_data_train, columns=num_df_train.columns, index=X_train.index)

X_train.drop(columns=scaled_num_df_train.columns, inplace=True)
X_test.drop(columns=scaled_num_df_test.columns, inplace = True)

X_test = pd.concat([scaled_num_df_test, X_test], axis = 1)
X_train = pd.concat([scaled_num_df_train, X_train], axis=1)

y_test = le.transform(y_test)


model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

st.write("### Model Accuracy and Precision")
st.write("Accuracy:", accuracy)
st.write("Precision:", precision)



st.write("### Model Performance Metrics")
st.write("The following metrics evaluate the performance of the model in predicting fraudulent claims:")
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Buat plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap="Blues")

# Tampilkan di Streamlit
st.pyplot(fig)

st.write("### Classification Report (Precision, Recall, F1-Score)")
st.dataframe(report)

# Atau tampilkan dengan format teks biasa
st.text(classification_report(y_test, y_pred))



st.subheader("XGBoost GridSearchCV")

if st.button("Run Grid Search"):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True)

    # Grid parameter
    param_grid = {
            'max_depth': [7, 10],
            'learning_rate': [0.1],
            'n_estimators': [100],
        }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    st.write("Best Parameters:", grid_search.best_params_)

        # ----------------------------
        # Evaluate
        # ----------------------------
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {acc:.4f}")
    precision = precision_score(y_test, y_pred)
    st.write(f"Precision: {precision:.4f}")