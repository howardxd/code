import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from binning_helper import BinningHelper
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App - ANN Model",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the ANN model and preprocessing components
@st.cache_resource
def load_model_components():
    """Load ANN model and preprocessing components"""
    try:

        model = load_model('models/ann_model.keras')
        
        # Load preprocessing components
        scaler = joblib.load('models/scaler.pkl')
        label_mappings = joblib.load('models/label_mappings.pkl')
        ordinal_mappings = joblib.load('models/ordinal_mappings.pkl')
        
        return model, scaler, label_mappings, ordinal_mappings
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

model, scaler, label_mappings, ordinal_mappings = load_model_components()

# Function to preprocess new data the same way as training data
def preprocess_for_prediction(input_data):
    """Transform raw inputs to match model features"""
    if model is None:
        st.error("Model not loaded properly")
        return None
        
    processed_data = input_data.copy()
    
    # Apply mappings to categorical features
    categorical_mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Smoking': {'Yes': 1, 'No': 0},
        'Family Heart Disease': {'Yes': 1, 'No': 0},
        'Diabetes': {'Yes': 1, 'No': 0},
        'High Blood Pressure': {'Yes': 1, 'No': 0},
        'Fasting Blood Sugar': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in processed_data:
            processed_data[col] = mapping[processed_data[col]]
    
    # Apply ordinal mappings for categories with natural order
    ordinal_categories = ['Exercise Habits', 'Stress Level', 'Sugar Consumption', 'Alcohol Consumption']
    
    for col in ordinal_categories:
        if col in processed_data and col in ordinal_mappings:
            mapping = {value: idx for idx, value in enumerate(ordinal_mappings[col])}
            processed_data[col] = mapping.get(processed_data[col], 0)
    
    # Create derived features
    if 'Age' in processed_data:
        age_group_label = BinningHelper.bin_age(processed_data['Age'])
        age_group_map = {value: idx for idx, value in enumerate(ordinal_mappings['Age Group'])}
        processed_data['Age Group'] = age_group_map[age_group_label]
        del processed_data['Age']  # Remove raw age as model uses Age Group
    
    # Process sleep hours
    if 'Sleep Hours' in processed_data:
        processed_data['Sleep Index'] = BinningHelper.index_sleep(processed_data['Sleep Hours'])
        del processed_data['Sleep Hours']  # Remove raw sleep hours as model uses Sleep Index

    # These are the EXACT columns in the EXACT order that the model expects
    expected_columns = [
        'Gender', 'Blood Pressure', 'Exercise Habits', 'Smoking',
        'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
        'Alcohol Consumption', 'Stress Level', 'Sugar Consumption',
        'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level',
        'Homocysteine Level', 'Sleep Index', 'Age Group', 'Cholesterol Factor'
    ]
    
    # Create a dictionary to hold values for the final dataframe
    final_data = {}
    
    # Fill in values from processed_data where they exist
    for col in expected_columns:
        if col in processed_data:
            final_data[col] = processed_data[col]
        else:
            # Provide default values for missing columns
            if col == 'BMI':
                final_data[col] = 22.0  # Normal BMI
            elif col == 'CRP Level':
                final_data[col] = 1.0  # Normal CRP
            elif col == 'Homocysteine Level':
                final_data[col] = 10.0  # Normal homocysteine
            elif col == 'Triglyceride Level':
                final_data[col] = 150.0  # Normal triglyceride
            elif col == 'Sleep Index':
                final_data[col] = 1  # Normal sleep index
            elif col == 'Age Group':
                final_data[col] = 2  # Adult
            elif col == 'Cholesterol Factor':
                final_data[col] = 1  # Normal cholesterol factor
            else:
                final_data[col] = 0  # Default for other columns
    
    # Create DataFrame with a single row containing the data
    final_df = pd.DataFrame([final_data])
    
    # Print for debugging
    st.write("Data being sent to ANN model:")
    st.write(final_df)
    
    # Scale numerical features
    scaled_data = scaler.transform(final_df)
    
    return scaled_data

def make_ann_prediction(model, preprocessed_data):
    """Make prediction using ANN model"""
    try:
        # Handle different model types
        if hasattr(model, 'predict_proba'):
            # KerasClassifier case
            prediction = model.predict(preprocessed_data)[0]
            prediction_proba = model.predict_proba(preprocessed_data)[0]
        else:
            # Direct Keras model case
            prediction_proba = model.predict(preprocessed_data)[0]
            prediction = 1 if prediction_proba[0] > 0.5 else 0
            
            # For direct Keras model, prediction_proba might be a single value
            if len(prediction_proba.shape) == 0 or prediction_proba.shape[0] == 1:
                # Binary classification with sigmoid output
                prob_heart_disease = float(prediction_proba)
                prob_no_heart_disease = 1.0 - prob_heart_disease
                prediction_proba = np.array([prob_no_heart_disease, prob_heart_disease])
            
        return prediction, prediction_proba
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Main app
def main():
    st.title("❤️ Heart Disease Prediction App - ANN Model")

    
    # Create a form for user input
    with st.form("prediction_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.radio("Gender", options=["Male", "Female"])
            age = st.number_input("Age", min_value=1, max_value=100, value=45)
            blood_pressure = st.slider("Blood Pressure (systolic)", min_value=90, max_value=200, value=120)
            
        with col2:
            family_history = st.radio("Family History of Heart Disease", options=["Yes", "No"])
            diabetes = st.radio("Do you have Diabetes?", options=["Yes", "No"])
            sleep_hours = st.slider("Sleep Hours per Night", min_value=3, max_value=12, value=7)
            
        st.subheader("Body Measurements")
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        with col2:
            height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170)
        bmi = weight / ((height/100)**2)
        st.info(f"Calculated BMI: {bmi:.1f}")
            
        st.subheader("Blood Test Results")
        col1, col2 = st.columns(2)
        with col1:
            high_blood_pressure = st.radio("Do you have High Blood Pressure?", options=["Yes", "No"])
            fasting_blood_sugar = st.radio("Fasting Blood Sugar > 120 mg/dl?", options=["Yes", "No"])
            
        with col2:
            triglyceride_level = st.slider("Triglyceride Level (mg/dL)", min_value=50, max_value=500, value=150)
            cholesterol_factor = st.slider("Cholesterol Factor", min_value=0, max_value=4, value=2, 
                                         help="Combined measurement of cholesterol levels (higher is worse)")
            
        st.subheader("Additional Blood Test Results")
        col1, col2 = st.columns(2)
        with col1:
            crp_level = st.slider("CRP Level (mg/L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                                help="C-Reactive Protein: marker of inflammation")
        with col2:
            homocysteine_level = st.slider("Homocysteine Level (μmol/L)", min_value=5.0, max_value=20.0, value=10.0, step=0.1,
                                      help="Amino acid in the blood - high levels are linked to heart disease")
            
        st.subheader("Lifestyle Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            smoking = st.radio("Do you smoke?", options=["Yes", "No"])
            
            exercise_options = ["High", "Medium", "Low"]
            exercise_habits = st.selectbox("Exercise Habits", options=exercise_options,
                                      help="High: >3hrs/week, Medium: 1-3hrs/week, Low: <1hr/week")
            
        with col2:
            stress_options = ["Low", "Medium", "High"]
            stress_level = st.selectbox("Stress Level", options=stress_options)
            
            alcohol_options = ["None", "Low", "Medium", "High"]
            alcohol_consumption = st.selectbox("Alcohol Consumption", options=alcohol_options,
                                          help="None: 0 drinks/week, Low: 1-2 drinks/week, Medium: 3-7 drinks/week, High: >7 drinks/week")
            
        sugar_options = ["Low", "Medium", "High"]
        sugar_consumption = st.selectbox("Sugar Consumption", options=sugar_options,
                                    help="Low: <25g/day, Medium: 25-50g/day, High: >50g/day")
            
        submitted = st.form_submit_button("🧠 Predict with ANN Model")
    
    if submitted:
        # Create a dictionary with the user inputs
        user_data = {
            'Gender': gender,
            'Age': age,
            'Blood Pressure': blood_pressure,
            'Family Heart Disease': family_history,
            'Diabetes': diabetes,
            'Sleep Hours': sleep_hours,
            'BMI': bmi,
            'Sugar Consumption': sugar_consumption,
            'CRP Level': crp_level,
            'Homocysteine Level': homocysteine_level,
            'Triglyceride Level': triglyceride_level,
            'High Blood Pressure': high_blood_pressure,
            'Fasting Blood Sugar': fasting_blood_sugar,
            'Cholesterol Factor': cholesterol_factor,
            'Exercise Habits': exercise_habits,
            'Smoking': smoking,
            'Alcohol Consumption': alcohol_consumption,
            'Stress Level': stress_level
        }
        
        # Preprocess the data
        preprocessed_data = preprocess_for_prediction(user_data)
        
        if preprocessed_data is not None:
            # Make prediction using ANN
            prediction, prediction_proba = make_ann_prediction(model, preprocessed_data)
            
            if prediction is not None and prediction_proba is not None:
                # Extract probabilities for both classes
                prob_no_heart_disease = prediction_proba[0]  # Probability of No Heart Disease (class 0)
                prob_heart_disease = prediction_proba[1]     # Probability of Heart Disease (class 1)
                
                # Display the results
                st.header("🧠 ANN Model Prediction Result")
                
                # Main prediction result
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction == 1:
                        st.error("⚠️ Higher Risk of Heart Disease Detected")
                    else:
                        st.success("✓ Lower Risk of Heart Disease Detected")
                
                with col2:
                    st.info(f"**ANN Prediction:** {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
                
                # Display both probabilities prominently
                st.subheader("Neural Network Prediction Probabilities")
                
                prob_col1, prob_col2, prob_col3 = st.columns([1, 1, 1])
                
                with prob_col1:
                    st.metric(
                        label="💚 No Heart Disease", 
                        value=f"{prob_no_heart_disease:.1%}",
                        help="ANN probability of NOT having heart disease"
                    )
                
                with prob_col2:
                    st.metric(
                        label="❤️ Heart Disease", 
                        value=f"{prob_heart_disease:.1%}",
                        help="ANN probability of having heart disease"
                    )
                
                # Enhanced visualization
                st.subheader("Visual Risk Assessment")
                
                # Create side-by-side charts
                viz_col1, viz_col2 = st.columns([1, 1])
                
                with viz_col1:
                    # Horizontal bar chart
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    
                    categories = ['No Heart Disease', 'Heart Disease']
                    probabilities = [prob_no_heart_disease, prob_heart_disease]
                    colors = ['lightgreen', 'lightcoral']
                    
                    bars = ax1.barh(categories, probabilities, color=colors)
                    ax1.set_xlim(0, 1)
                    ax1.set_xlabel('Probability')
                    ax1.set_title('ANN Model: Heart Disease Risk Probabilities')
                    
                    # Add percentage labels on bars
                    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                        ax1.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                                f'{prob:.1%}', ha='center', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig1)
                
                with viz_col2:
                    # Pie chart
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    
                    labels = ['No Heart Disease', 'Heart Disease']
                    sizes = [prob_no_heart_disease, prob_heart_disease]
                    colors = ['lightgreen', 'lightcoral']
                    explode = (0.1, 0) if prediction == 0 else (0, 0.1)  # Explode the predicted class
                    
                    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                                      colors=colors, explode=explode, shadow=True, startangle=90)
                    
                    # Make percentage text bold
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(12)
                    
                    ax2.set_title('ANN Model: Heart Disease Risk Distribution', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig2)
                
                # Risk level indicator
                st.subheader("Risk Level Assessment")
                
                if prob_heart_disease < 0.3:
                    risk_level = "LOW RISK"
                    risk_color = "success"
                    risk_message = "The neural network indicates low probability of heart disease. Continue maintaining healthy habits!"
                elif prob_heart_disease < 0.6:
                    risk_level = "MODERATE RISK"
                    risk_color = "warning"
                    risk_message = "The neural network indicates moderate risk. Consider lifestyle improvements and regular check-ups."
                else:
                    risk_level = "HIGH RISK"
                    risk_color = "error"
                    risk_message = "The neural network indicates high risk. Please consult with a healthcare professional immediately."
                
                if risk_color == "success":
                    st.success(f"**{risk_level}**: {risk_message}")
                elif risk_color == "warning":
                    st.warning(f"**{risk_level}**: {risk_message}")
                else:
                    st.error(f"**{risk_level}**: {risk_message}")
                
                # Detailed probability breakdown
                with st.expander("🧠 Detailed ANN Model Analysis"):
                    st.write("**Neural Network Prediction Details:**")
                    st.write(f"• Probability of **No Heart Disease**: {prob_no_heart_disease:.4f} ({prob_no_heart_disease:.2%})")
                    st.write(f"• Probability of **Heart Disease**: {prob_heart_disease:.4f} ({prob_heart_disease:.2%})")
                    st.write(f"• **Final Prediction**: {prediction} ({'Heart Disease' if prediction == 1 else 'No Heart Disease'})")
                    st.write(f"• **Decision Threshold**: 0.5 (50%)")
                    st.write(f"• **Model Confidence**: {max(prob_no_heart_disease, prob_heart_disease):.2%}")
                    
                    if prob_heart_disease > 0.5:
                        st.write(f"• Since heart disease probability ({prob_heart_disease:.2%}) > 50%, the ANN predicts **Heart Disease**")
                    else:
                        st.write(f"• Since heart disease probability ({prob_heart_disease:.2%}) < 50%, the ANN predicts **No Heart Disease**")
                
                # Show risk factors based on input
                with st.expander("Your Risk Factors"):
                    risk_factors = []
                    if user_data['Gender'] == 'Male':
                        risk_factors.append("Men generally have a higher risk of heart disease.")
                    if age > 45:
                        risk_factors.append("Age above 45 increases heart disease risk.")
                    if blood_pressure > 130:
                        risk_factors.append("Elevated blood pressure is a significant risk factor.")
                    if family_history == "Yes":
                        risk_factors.append("Family history of heart disease increases your risk.")
                    if diabetes == "Yes":
                        risk_factors.append("Diabetes significantly increases heart disease risk.")
                    if cholesterol_factor > 2:
                        risk_factors.append("Your cholesterol profile increases your risk.")
                    if smoking == "Yes":
                        risk_factors.append("Smoking significantly increases heart disease risk.")
                    if sleep_hours < 6 or sleep_hours > 9:
                        risk_factors.append("Suboptimal sleep patterns can impact heart health.")
                    if bmi >= 25:
                        risk_factors.append(f"Your BMI of {bmi:.1f} indicates overweight, which increases risk.")
                    if sugar_consumption == "High":
                        risk_factors.append("High sugar consumption can contribute to heart disease risk.")
                    if exercise_habits == "Low":
                        risk_factors.append("Insufficient physical activity increases risk.")
                    if stress_level == "High":
                        risk_factors.append("High stress levels may contribute to heart problems.")
                    if alcohol_consumption in ["Medium", "High"]:
                        risk_factors.append("Moderate to high alcohol consumption may increase risk.")
                    if crp_level > 3:
                        risk_factors.append("Elevated CRP levels indicate inflammation, a risk factor.")
                    if homocysteine_level > 15:
                        risk_factors.append("High homocysteine levels are associated with increased heart disease risk.")
                    if triglyceride_level > 150:
                        risk_factors.append("Elevated triglyceride levels can contribute to heart disease.")
                    if high_blood_pressure == "Yes":
                        risk_factors.append("High blood pressure is a major risk factor for heart disease.")
                    if fasting_blood_sugar == "Yes":
                        risk_factors.append("Elevated fasting blood sugar may indicate prediabetes or diabetes, increasing risk.")
                        
                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(factor)
                    else:
                        st.info("No significant risk factors identified from your inputs.")
                
                # Recommendations section
                st.subheader("Recommendations")

                if prediction == 0:  # No Heart Disease
                    st.markdown("""
                    ### Steps to maintain heart health:
                    
                    * **Regular exercise**: Continue at least 150 minutes of moderate activity weekly
                    * **Balanced diet**: Focus on fruits, vegetables, whole grains, and lean proteins
                    * **Monitor regularly**: Keep track of blood pressure, cholesterol, and glucose levels
                    * **Manage stress**: Practice relaxation techniques like meditation or yoga
                    * **Quality sleep**: Maintain 7-9 hours of quality sleep nightly
                    * **Avoid smoking**: Continue to avoid tobacco products
                    * **Limit alcohol**: Keep alcohol consumption moderate
                    * **Regular check-ups**: Schedule annual wellness visits with your healthcare provider
                    
                    Always consult with healthcare professionals for personalized medical advice.
                    """)

                else:  # Heart Disease Risk Detected
                    st.markdown("""
                    ### Immediate steps to reduce heart disease risk:
                    
                    * **Consult a doctor immediately**: Schedule an appointment with a cardiologist or primary care physician
                    * **Medication management**: Discuss current medications and potential cardiac treatments
                    * **Heart-healthy diet**: Adopt a DASH or Mediterranean diet with low sodium
                    * **Increase physical activity**: Start with gentle exercise like walking (with doctor approval)
                    * **Stop smoking**: If you smoke, quit immediately and seek professional help
                    * **Limit alcohol**: Reduce alcohol consumption significantly
                    * **Stress management**: Practice daily stress reduction techniques
                    * **Monitor vital signs**: Check blood pressure and other metrics regularly
                    * **Weight management**: Work toward achieving and maintaining a healthy weight
                    
                    **Important**: These recommendations do not replace professional medical advice. Please consult with healthcare professionals immediately for proper evaluation and treatment.
                    """)

if __name__ == "__main__":
    main()