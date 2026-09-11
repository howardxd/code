===============================================================================
                 HEART DISEASE PREDICTION APP
                 Using Artificial Neural Networks
===============================================================================

OVERVIEW
--------
This application predicts the likelihood of heart disease based on various health 
indicators and demographic factors. It uses a trained Artificial Neural Network 
(ANN) model to provide predictions and insights about heart disease risk factors.

FEATURES
--------
1. Interactive web interface built with Streamlit
2. Heart disease prediction using a trained ANN model
3. Data preprocessing and feature engineering
4. Visualization of prediction results
5. Patient data input through user-friendly forms
6. Analysis of key risk factors

FILES AND STRUCTURE
------------------
- app.py: Main Streamlit application file for the user interface
- binning_helper.py: Helper class for binning continuous variables (age, cholesterol, etc.)
- timer.py: Utility for performance timing measurements
- heart_disease.csv: Dataset containing patient records with various health metrics
- requirements.txt: List of Python dependencies needed to run the application
- heart_disease_analysis.ipynb: Jupyter notebook with data exploration and model development

Models directory:
- ann_model.keras: Trained Artificial Neural Network model
- scaler.pkl: Fitted MinMaxScaler for standardizing numerical features
- label_mappings.pkl: Encodings for categorical variables
- ordinal_mappings.pkl: Mappings for ordinal variables

INSTALLATION
-----------
1. Ensure you have Python 3.8+ installed
2. Clone or download this repository
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

USAGE
-----
1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Access the application in your web browser (typically at http://localhost:8501)
3. Input patient information in the provided form
4. View prediction results and risk factor analysis

DATA DESCRIPTION
---------------
The heart disease dataset includes the following features:
- Demographic: Age, Gender
- Clinical measurements: Blood Pressure, Cholesterol, BMI, Triglycerides, etc.
- Lifestyle factors: Exercise Habits, Smoking, Alcohol Consumption
- Medical history: Family Heart Disease, Diabetes
- Other health metrics: Stress Level, Sleep Hours, Sugar Consumption

MODEL INFORMATION
---------------
The prediction model is an Artificial Neural Network trained on the dataset to
classify patients as having or not having heart disease. The model processes both
categorical and numerical features that have been appropriately scaled and encoded.

DEVELOPMENT
----------
The Jupyter notebook (heart_disease_analysis copy.ipynb) contains the full development
process including:
- Data loading and exploration
- Feature engineering and preprocessing
- Model training and hyperparameter tuning
- Performance evaluation
- Comparison with other algorithms (Random Forest, XGBoost, etc.)

PERFORMANCE
----------
The model achieves good predictive performance with metrics evaluated on a test set.
Refer to the notebook for detailed performance analysis including accuracy, precision,
recall, and F1-score.

DEPENDENCIES
-----------
Main dependencies include:
- TensorFlow/Keras: Deep learning framework
- Streamlit: Web application framework
- Pandas & NumPy: Data manipulation
- Scikit-learn: Machine learning utilities
- Matplotlib & Seaborn: Visualization

ACKNOWLEDGMENTS
--------------
This project was developed as a demonstration of applying deep learning techniques
to healthcare data for predictive analytics.

===============================================================================