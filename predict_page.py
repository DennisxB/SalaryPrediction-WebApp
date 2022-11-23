import streamlit as st
import numpy as np
import pickle 


# load model
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


# user interface 
def show_predict_page():
    st.title("Software Developer Salary Prediction")
    
    st.write("""### We Need Some Information To to Predict""")
   
   # select box
    countries = (
        'United States of America',                                
        'Germany',                                              
        'United Kingdom of Great Britain and Northern Ireland',    
        'India',                                                   
        'Canada',                                                 
        'France',                                                  
        'Brazil',                                                  
        'Spain',                                                    
        'Netherlands',                                             
        'Australia',                                                
        'Italy',                                                    
        'Poland',                                                   
        'Sweden',                                                   
        'Russian Federation',                                       
        'Switzerland'
    )

    education = (
        'Less than Bachelor’s degree', 
        'Bachelor’s degree', 'Master’s degree', 
        'Post Grad.'
    )

    country = st.selectbox("Countries", countries)
    education_level = st.selectbox("Education Level", education)

    # experience
    experience = st.slider("Years of Experience", 0, 50, 3)

    # prediction button
    ok = st.button("Calculate Salary")
    if ok:
        new_data = np.array([[country, education_level, experience]])
        new_data[:, 0] = le_country.transform(new_data[:, 0])
        new_data[:, 1] = le_education.transform(new_data[:, 1])
        new_data = new_data.astype(float)

        # prediction
        salary = regressor_loaded.predict(new_data)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
