import pickle
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Failure Classifier",
    page_icon="images/icon.png",
    )

# Page title
st.title('Maintenance - Failure Prediction')
st.image('images/icon.png')
st.write("\n\n")

st.markdown(
    """
    This app aims to assist in classifying failures, thereby reducing the time required to analyze machine problems. 
    It enables the analysis of sensor data to classify failures swiftly and expedite the troubleshooting process.    
    """
)

# Load the Model
with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
# Streamlit inteface to input data
col1, col2 = st.columns(2)

with col1:
    air = st.number_input(label='Air Temperature')
    process = st.number_input(label='Process Temperature')
    rpm = st.number_input(label='Rotational Speed')

with col2:
    torque = st.number_input(label='Torque')
    tool_wear = st.number_input(label='Tool Wear')
    type = st.selectbox(label='Type', options=['Low','Medium','High'])
    
def prediction(air, process, rpm, torque, tool_wear, type):
    
    df_input = pd.DataFrame({
    "Air_temperature":[air],
    "Process_temperature":[process],
    "Rotational_speed":[rpm],
    "Torque":[torque],
    "Tool_wear":[tool_wear],
    "Type":[type],        
    })
    
    prediction = model.predict(df_input)
    return prediction

# Bottom to Predict
if st.button('Predict'):
    predict = prediction(air, process, rpm, torque, tool_wear, type)
    st.success(predict)