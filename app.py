import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
from PIL import Image

# Load the pre-trained deep learning model
model = tf.keras.models.load_model('employee_attrition_model.h5')

# Load the full preprocessing pipeline using pickle
with open('employeeAttrition_pred.pkl', 'rb') as f:
    full_pipeline = pickle.load(f)

# Define the prediction function
def predict_attrition(data):
    # Preprocess the input data using the full pipeline
    input_data = full_pipeline.transform(data)
    
    # Make predictions using the loaded model
    predictions = model.predict(input_data)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    return predicted_classes

# Function to display image
def display_image(image_path):
    image = Image.open(image_path)
    st.image(image, caption='', use_column_width=True)

def main():
    # Set app title
    st.title('StayOrGo')
    # Display an image
    image_path = 'employee_attrition.jpg'
    display_image(image_path)
    # Add app description
    st.write("""
        Welcome to StayOrGo! An Employee Attrition Prediction App!
        Use this app to predict whether an employee is likely to leave the company or not.
        Please provide some information about the employee to get started.
        """)

    # Create input widgets for user input - Gender and Age
    # st.subheader('Employee General Information:')
    if 'Confirmed' not in st.session_state:
        st.session_state['Confirmed'] = False
    if 'Age' not in st.session_state:
        st.session_state['Age'] = None

    age = st.number_input('Age', min_value=18, max_value=60, value=30)

    # Store the user inputs in the session state
    st.session_state['Age'] = age

    # Create a confirm button for the gender and age inputs
    confirm_button = st.button("Confirm")

    # Check if the confirm button is clicked
    if confirm_button:
        st.session_state['Confirmed'] = True

    # Check if the gender, age, and confirmation have been provided
    if age is not None and st.session_state['Confirmed']:
        st.write(f"Confirmed age: {age}")
    else:
        st.write("Please confirm your age before proceeding.")

    with st.sidebar:
        job_level = st.sidebar.slider('Job Level', 1, 5, 3)
        total_working_years = st.sidebar.slider('Total Working Years', 0, 42, 10)
        years_in_current_role = st.sidebar.slider('Years in Current Role', 0, 42, 3)
        monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=200000, value=5000, step=100)
        job_role = st.sidebar.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                                'Sales Representative', 'Research Director', 'Human Resources'], index=0)
        marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'], index=0)
        over_time = st.sidebar.radio('Over Time', ['Yes', 'No'], index=0)

        # Create a predict button
        predict_button = st.sidebar.button("Predict")

    if predict_button:
        # Create a dictionary with user inputs
        input_data = {
            'Age': [age],
            'JobLevel': [job_level],
            'TotalWorkingYears': [total_working_years],
            'YearsInCurrentRole': [years_in_current_role],
            'MonthlyIncome': [monthly_income],
            'JobRole': [job_role],
            'MaritalStatus': [marital_status],
            'OverTime': [over_time],
        }

        # Create a DataFrame from the input data
        df_input = pd.DataFrame(input_data)

        # Get predictions for the input data
        predictions = predict_attrition(df_input)
        df_input_vertical = df_input.T

        # Display the DataFrame with the input data vertically
        st.header("Input Data")
        st.table(df_input_vertical)

        # Display the prediction result
        if predictions[0] == 0:
            st.header("Result")
            st.subheader("The employee is not likely to leave.")
            st.write("Great news! The employee is satisfied with their job.")
        else:
            st.header("Result")
            st.subheader("The employee is likely to leave.")
            image_path2 = 'Tips-reduce-employee-turnover-retain-top-performers-1.png'
            st.write("Take necessary steps to retain the employee.")
            display_image(image_path2)

if __name__ == '__main__':
    main()

