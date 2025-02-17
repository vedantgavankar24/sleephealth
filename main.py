import streamlit as st
import pandas as pd
import pickle
import numpy as np
import lightgbm
import requests
from transformers import pipeline
from huggingface_hub import login
login("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")

def recommendation(prediction, prompt, recommendations):
    st.subheader('Recommendations')
    st.info("This application serves as a tool to aid medical professionals in diagnosing conditions, however, it should not be relied upon as a replacement for professional diagnosis.")
    st.markdown(recommendations)

def sidebar():
    st.sidebar.header('Parameters')
    st.sidebar.write('Please put the correct value in each')

    g = ['Male', 'Female']
    g1 = [1, 0]
    mapping_gender = dict(zip(g, g1))
    select_gender = st.sidebar.selectbox("Gender", g)
    mapping1 = mapping_gender[select_gender]

    o = ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher',
         'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer',
         'Salesperson', 'Manager']
    o1 = [9, 1, 6, 10, 5, 2, 0, 8, 3, 7, 4]
    mapping_Occupation = dict(zip(o, o1))
    select_Occupation = st.sidebar.selectbox("Occupation", o)
    mapping3 = mapping_Occupation[select_Occupation]

    sleep_duration = st.sidebar.slider('Hours of Sleep Duration (per day)', min_value=1, max_value=24)
    quality = st.sidebar.slider('Quality of sleep (scale: 1-10)', min_value=1, max_value=10)
    physical_activity = st.sidebar.slider('Physical Activity Level (minutes/day)', min_value=0, max_value=300)
    stress = st.sidebar.slider('Stress Level (scale: 1-10)', min_value=1, max_value=10)

    b = ['Normal', 'Obese', 'Overweight']
    b1 = [0, 1, 2]
    mapping_BMI = dict(zip(b, b1))
    select_BMI = st.sidebar.selectbox("BMI Category", b)
    mapping4 = mapping_BMI[select_BMI]

    heart_rate = st.sidebar.slider('Heart rate (bpm)', min_value=60, max_value=170)
    daily_steps = st.sidebar.slider('Number of steps you take (per day)', min_value=0, max_value=10000)

    a = ['26-35', '36-45', '46-55', '55+']
    a1 = [0, 1, 2, 3]
    mapping_age = dict(zip(a, a1))
    select_age = st.sidebar.selectbox("Age", a)
    mapping2 = mapping_age[select_age]

    c = ['Elevated', 'High (Stage 1)', 'High (Stage 2)', 'Normal']
    c1 = [0, 1, 2, 3]
    mapping_bp = dict(zip(c, c1))
    select_bp = st.sidebar.selectbox("Blood Pressure Level", c)
    mapping5 = mapping_bp[select_bp]

    df_for_pred = pd.DataFrame({
        'Gender': mapping1,
        'Occupation': mapping3,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress,
        'BMI Category': mapping4,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'Age': mapping2,
        'Blood Pressure': mapping5
    }, index=[0])

    st.write(df_for_pred)
    return df_for_pred

def prediction(df):
    try:
        model = pickle.load(open('lgbm_model.pkl', 'rb'))  # Load the LightGBM model
        prediction = model.predict(df)
        prediction_class = int(np.argmax(prediction, axis=1)) if len(prediction.shape) > 1 else int(prediction[0] > 0.5)
        probs = prediction[0] if len(prediction.shape) > 1 else [1 - prediction[0], prediction[0]]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    col1, col2 = st.columns([3, 3])
    pred = ['Healthy', 'Insomnia', 'Sleep Apnea'][prediction_class]
    
    with col1:
        col1.subheader(pred)
        if len(probs) > 2:
            st.write("Probability of Healthy: ", probs[0])
        st.write("Probability of Insomnia: ", probs[1])
        st.write("Probability of Sleep Apnea: ", probs[2])

        st.subheader("Chat with the Assistant")
        user_input = st.text_input("Ask a question:")
        
        if st.button("Send"):
            response = simulate_chatbot_response(user_input)
            st.markdown(f"Assistant: {response}")
    
    with col2:
        # Reverse mapping dictionaries
        reverse_mapping_gender = {1: 'Male', 0: 'Female'}
        reverse_mapping_occupation = {
            9: 'Software Engineer', 1: 'Doctor', 6: 'Sales Representative', 10: 'Teacher',
            5: 'Nurse', 2: 'Engineer', 0: 'Accountant', 8: 'Scientist', 3: 'Lawyer',
            7: 'Salesperson', 4: 'Manager'
        }
        reverse_mapping_age = {0: '26-35', 1: '36-45', 2: '46-55', 3: '55+'}
        reverse_mapping_BMI = {0: 'Normal', 1: 'Obese', 2: 'Overweight'}
        reverse_mapping_bp = {0: 'Elevated', 1: 'High (Stage 1)', 2: 'High (Stage 2)', 3: 'Normal'}

        # Convert integer values back to labels
        gender_label = reverse_mapping_gender[df.iloc[0]['Gender']]
        occupation_label = reverse_mapping_occupation[df.iloc[0]['Occupation']]
        age_label = reverse_mapping_age[df.iloc[0]['Age']]
        bmi_label = reverse_mapping_BMI[df.iloc[0]['BMI Category']]
        bp_label = reverse_mapping_bp[df.iloc[0]['Blood Pressure']]

        # Corrected Prompt
        prompt = f"""
        You are a medical assistant specializing in sleep disorders. Provide personalized recommendations for a user with:
        - Gender: {gender_label}
        - Occupation: {occupation_label}
        - Age Group: {age_label}
        - Diagnosis: {pred}
        - Sleep Duration: {df.iloc[0]['Sleep Duration']} hours/day
        - Quality of Sleep: {df.iloc[0]['Quality of Sleep']}/10
        - Physical Activity Level: {df.iloc[0]['Physical Activity Level']} minutes/day
        - Stress Level: {df.iloc[0]['Stress Level']}/10
        - BMI Category: {bmi_label}
        - Heart Rate: {df.iloc[0]['Heart Rate']} bpm
        - Daily Steps: {df.iloc[0]['Daily Steps']} steps/day
        - Blood Pressure Level: {bp_label}
        Provide a numbered list of 7 actionable recommendations.
        """

        try:
            response = get_hf_response(prompt)
            
            if isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
                recommendations = response[0]["generated_text"][len(prompt):].strip()
            else:
                recommendations = "No valid response received from the API."
            if recommendations:
                lines = recommendations.split('\n')
                recommendations = '\n'.join([line.strip() for line in lines])
        except Exception as e:
            st.error(f"Failed to generate recommendations: {e}")
            recommendations = ""


        
        recommendation(prediction_class, prompt, recommendations)

def get_hf_response(prompt):
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

    headers = {"Authorization": "Bearer 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"}
            
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

    return response.json()

def simulate_chatbot_response(user_message):
    try:
        response = get_hf_response(user_message)  # Fixed the extra parenthesis
        response_data = response.json()
        
        if isinstance(response_data, list) and len(response_data) > 0:
            return response_data
        else:
            return "Sorry, I couldn't understand that."
    
    except Exception as e:
        return f"Error: {e}"


def main():
    st.set_page_config(page_title="Sleep Disorder Prediction", page_icon="😴", layout="wide")
    st.title('Sleep Disorder Prediction😴')
    st.write('Assess and predict the likelihood of sleep disorders using machine learning.')
    df_pred = sidebar()
    if st.button("Click for Result"):
        prediction(df_pred)


if __name__ == '__main__':
    main()
