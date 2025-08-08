import streamlit as st
import pandas as pd
import numpy as np
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
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Model Performance",
    page_icon=":bar_chart:",
    layout="wide",
)
st.sidebar.image("andito_l_50371.jpg", width=100)  # Replace with your logo URL

st.title("Prediction")
st.subheader("Model prediction")
st.write("This section demonstrates how to use the trained model to make predictions on new data. It includes input fields for the user to enter feature values and a button to generate predictions.")


df = pd.read_csv("C:/Users/gito2/Downloads/streamlit-demo-app-main/insurance_claims.csv")  # Assuming you have a CSV file with your projects
df.drop(["incident_hour_of_the_day", 'insured_zip', 'policy_bind_date', 'incident_location'], axis=1, inplace=True)


x= df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]

 # Display the count of fraud and non-fraud cases

cat_df = x.select_dtypes(include = ['object'])

fraud_count = y.value_counts()


#fraud_reported vs other features

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
            'learning_rate': [0.1, 0.01],
            'n_estimators': [100, 200],
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