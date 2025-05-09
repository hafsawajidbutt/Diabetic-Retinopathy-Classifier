# import streamlit as st
# import numpy as np
# import time
# from PIL import Image
# import matplotlib.pyplot as plt
# from use_dr_model import predict_image, load_pretrained_model
# # Set page configuration
# st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon="🩺", layout="centered")

# #CSS for colors
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #FFF7F4; /* Lightest shade */
#     }
#     .stProgress > div > div > div > div {
#         background-color: #EAB8B8 !important; /* Mid-tone shade */
#     }
#     div.stButton > button:first-child {
#         background-color: #C599B6; /* Main theme color */
#         color: white;
#         border-radius: 10px;
#         padding: 10px;
#         font-size: 16px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Title
# st.markdown("<h1 style='color: #C599B6; text-align: center;'>Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)

# # File uploader
# uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     temp_file_path = "temp_image.jpg"
#     # Save the uploaded file to a temporary location
#     with open(temp_file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     # Now you can use the file path
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
    
#     # Processing indicator
#     with st.spinner("Analyzing image..."):
#         model = load_pretrained_model("./model/diabetic_retinopathy_cnn_model.pth")
#         print(image)
#         predicted_class, predicted_percentages = predict_image(model, temp_file_path)
#         print("Predicted Class: ", predicted_class)
#         print("Predicted Percentages: ", predicted_percentages)
#         average = np.mean(predicted_percentages)
#         severity_percentage = average 
    
#     # Display result
#     st.markdown(f"<h2 style='color: #D8A0A8;'>Severity Level: {severity_percentage}%</h2>", unsafe_allow_html=True)
#     st.progress(severity_percentage / 100)  #progress bar
    
#     #severity as a pie chart
#     fig, ax = plt.subplots()
#     ax.pie([severity_percentage, 100 - severity_percentage], labels=["DR Severity", "Healthy"], colors=["#EAB8B8", "#FFD1C1"], autopct='%1.1f%%')
#     st.pyplot(fig)
    
#     #reprocess button
#     if st.button("Analyze Another Image"):
#         st.rerun()
import streamlit as st
import numpy as np
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from use_dr_model import predict_image, load_pretrained_model

# Set page configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="🩺",
    layout="centered"
)

# CSS for colors
st.markdown(
    """
    <style>
    body {
        background-color: #FFF7F4; /* Lightest shade */
    }
    .stProgress > div > div > div > div {
        background-color: #EAB8B8 !important; /* Mid-tone shade */
    }
    div.stButton > button:first-child {
        background-color: #C599B6; /* Main theme color */
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='color: #C599B6; text-align: center;'>Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)

# Check model path
MODEL_PATH = "./model/diabetic_retinopathy_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path and ensure the model file exists.")
    st.info(f"Current working directory: {os.getcwd()}")
    
    # Check if the model directory exists
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        st.warning(f"Model directory '{model_dir}' does not exist. Creating it now.")
        try:
            os.makedirs(model_dir, exist_ok=True)
            st.success(f"Created directory: {model_dir}")
        except Exception as e:
            st.error(f"Failed to create directory: {e}")
    
    # List files in current directory for debugging
    st.write("Files in current directory:", os.listdir("."))
    
    if model_dir and os.path.exists(model_dir):
        st.write(f"Files in model directory:", os.listdir(model_dir))
    
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    temp_file_path = "temp_image.jpg"
    
    # Save the uploaded file to a temporary location
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Now you can use the file path
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Processing indicator
    with st.spinner("Analyzing image..."):
        # Load model (with better error handling)
        model = load_pretrained_model(MODEL_PATH)
        
        if model is None:
            st.error("Failed to load the model. Please check the console for error details.")
            st.stop()
        
        # Make prediction
        predicted_class, severity_percentage = predict_image(model, temp_file_path)
        
        if predicted_class == "Error":
            st.error("An error occurred during prediction. Please try another image or check the logs.")
            st.stop()
            
        st.success("Analysis complete!")
    
    # Display result
    st.markdown(f"<h2 style='color: #D8A0A8;'>Diagnosis: {predicted_class}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #D8A0A8;'>Severity Level: {severity_percentage:.1f}%</h3>", unsafe_allow_html=True)
    st.progress(severity_percentage / 100)  # progress bar
    
    # Severity as a pie chart
    fig, ax = plt.subplots()
    ax.pie(
        [severity_percentage, 100 - severity_percentage], 
        labels=["DR Severity", "Healthy"], 
        colors=["#EAB8B8", "#FFD1C1"], 
        autopct='%1.1f%%'
    )
    st.pyplot(fig)
    
    # Recommendations based on severity
    st.subheader("Recommendations")
    
    if severity_percentage < 25:
        st.success("No immediate action required. Continue regular diabetes management and annual eye screenings.")
    elif severity_percentage < 50:
        st.info("Follow up with an ophthalmologist within 6 months. Monitor blood sugar levels closely.")
    elif severity_percentage < 75:
        st.warning("Consult with an ophthalmologist within 3 months. Tight control of blood sugar, blood pressure, and lipids is essential.")
    else:
        st.error("Urgent referral to an ophthalmologist is recommended. Prompt treatment may be necessary to prevent vision loss.")
    
    # Reprocess button
    if st.button("Analyze Another Image"):
        st.rerun()  # Updated from experimental_rerun()