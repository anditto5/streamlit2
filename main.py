import streamlit as st
import pandas as pd
import numpy as np
from utils import load_css

load_css()

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
st.write(" Hard skills = [Machine Learning, Deep Learning, Data Analysis, Model Deployment, Data Visualization]")
st.write("Soft skills = [Problem Solving, Critical Thinking, Communication, Teamwork, Adaptability, Time Management, Attention to Detail]")

st.write("## Tools and Technologies")
st.write("Python, Streamlit, Pandas, NumPy, Scikit-learn, TensorFlow, Keras, SQL, Git, Docker, Jupyter Notebook, Visual Studio Code")

st.write("## Languages")
st.write("English (Fluent), Bahasa Indonesia (Native), Japanese (basic)")


st.write("## Education")
st.write("Bachelor of Mechanical Engineering, Diponegoro University, 2014-2021, Semarang, Indonesia, GPA: 2.79/4.0")
st.write("Master of Data Science, Universitas Indonesia, 2022-2024, Depok, Indonesia, GPA: 3.19/4.0")

st.write("## Certifications")
st.write("•	Bootcamp Deep Learning for Medical Imaging" )
st.write("•	Rapid Data Science Mastery ")
st.write("•	Mastering python & EDA ( Unlocking Data Insight & with exploration data analysis )")

