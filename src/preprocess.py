import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# Download the stopwords list silently
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_message(message):
    """Removes punctuation, makes lowercase, and removes stopwords."""
    message = str(message).lower()
    message = message.translate(str.maketrans('', '', string.punctuation))
    
    words = message.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

def main():
    print("Loading raw dataset...")
    # Load the file from your main project folder
    df = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Drop empty columns and rename the important ones
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df.rename(columns={'v1': 'target', 'v2': 'text'})
    
    print("Cleaning text messages (this might take a few seconds)...")
    df['cleaned_text'] = df['text'].apply(clean_message)
    
    # Convert 'ham' and 'spam' into 0 and 1 so the math model can understand it
    df['label'] = df['target'].map({'ham': 0, 'spam': 1})
    
    # Save the cleaned data into your data folder
    df.to_csv('data/cleaned_spam.csv', index=False)
    print("Success! Cleaned data saved to data/cleaned_spam.csv")

# This ensures the code runs when we execute the file
if __name__ == "__main__":
    main()