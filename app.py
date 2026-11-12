import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# We need the exact same cleaning function used during training
def clean_message(message):
    message = str(message).lower()
    message = message.translate(str.maketrans('', '', string.punctuation))
    words = message.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Load the saved model and vectorizer (cached so it only loads once)
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("Model files not found! Please run `python src/train.py` first.")
    st.stop()

# --- WEB UI DESIGN ---
st.title("📱 SMS Spam Classifier")
st.write("Enter a text message below to see if the AI thinks it is Spam or Safe.")

# Text input box for the user
user_input = st.text_area("Message:", placeholder="Type a fake spam message here...", height=150)

# The analyze button
if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        # 1. Clean the input
        cleaned_input = clean_message(user_input)
        
        # 2. Convert words to numbers using the loaded vectorizer
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # 3. Make the prediction
        prediction = model.predict(vectorized_input)[0]
        
        # 4. Show the result
        if prediction == 1:
            st.error("🚨 **SPAM DETECTED!** Do not click any links.")
        else:
            st.success("✅ **SAFE!** This looks like a normal message.")