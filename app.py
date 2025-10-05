import streamlit as st
import joblib

#Load the pipeline
try :
    pipeline = joblib.load('spam_model.pkl')
    print("Model loaded successfully.")

except FileNotFoundError:
    st.error("Model file not found. Please run train.py first to create the model.")
    st.stop()

# Web interface 
#Set title of the app
#title
st.title('Spam Message Detector')
#input

user_input = st.text_area('Enter your message here to check if it is spam or ham:', height=150)


#Create button 

if st.button('Check Message'):
    if user_input:
        #pipeline handles both vectorizing and predicting
        prediction = pipeline.predict([user_input])[0]
        probability = pipeline.predict_proba([user_input])

        st.subheader('Result')
        if prediction =='spam':
            st.error('This message is classified as SPAM', icon="🚨")
            st.write(f"Spam Probability: {probability[0][1]:.2%}")
        else:
            st.success('This message is classified as HAM (not spam).', icon="✅")
            st.write(f"Ham Probability: {probability[0][0]:.2%}")
    else:
        st.warning('Please enter a message to check.')
            
''' 
#With more webpage interaction

# app.py
import streamlit as st
import joblib
import time

# --- PAGE CONFIGURATION ---
# This should be the first Streamlit command in your script
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- MODEL LOADING ---
# Use a function with st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('spam_model.pkl')
        return pipeline
    except FileNotFoundError:
        return None

pipeline = load_model()

# --- SIDEBAR ---
st.sidebar.header("About the Project")
st.sidebar.markdown("""
This application is a real-time Spam Message Detector built using a Naive Bayes machine learning model.
- **Model:** `MultinomialNB`
- **Vectorization:** `CountVectorizer`
- **Accuracy:** The model was trained on the SMS Spam Collection Dataset and achieved a high accuracy on the test set.
""")
st.sidebar.info("Enter a message in the text box and click 'Check Message' to see the prediction.")

# --- MAIN INTERFACE ---
st.title("Spam Message Detector 📧")
st.subheader("Enter a message to classify it as Spam or Ham (Not Spam)")

# Provide example buttons
st.write("Don't know what to test? Try one of these examples:")
col1, col2 = st.columns(2)
with col1:
    if st.button("Try a Spam Example"):
        st.session_state.user_input = "Congratulations! You've won a $1000 prize. Click here to claim."
with col2:
    if st.button("Try a Ham Example"):
        st.session_state.user_input = "Hi team, just a reminder about the project meeting tomorrow at 10am."

# The text area for user input, pre-filled by session state if a button was clicked
user_input = st.text_area(
    'Your message:',
    key='user_input', # Give it a key to access it with st.session_state
    height=150
)

# Prediction button
if st.button('Check Message', type="primary"):
    if pipeline is None:
        st.error("Model file not found. Please run `train.py` first to create the model.")
    elif user_input:
        # Show a spinner while predicting
        with st.spinner('Analyzing your message...'):
            time.sleep(1) # Simulate a short delay for effect

            # Predict and get probabilities
            prediction = pipeline.predict([user_input])[0]
            probability = pipeline.predict_proba([user_input])

            # --- DISPLAY RESULTS ---
            st.subheader('Result:')
            if prediction == 'spam':
                st.error('This message is classified as SPAM.', icon="🚨")
                spam_proba = probability[0][1]
                st.metric(label="Spam Confidence Score", value=f"{spam_proba:.2%}")

            else:
                st.success('This message is classified as HAM (not spam).', icon="✅")
                ham_proba = probability[0][0]
                st.metric(label="Ham Confidence Score", value=f"{ham_proba:.2%}")
    else:
        st.warning('Please enter a message to check.')
'''