import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import pickle

def main():
    print("1. Loading cleaned data...")
    df = pd.read_csv('data/cleaned_spam.csv')
    
    # Sometimes cleaning removes all words (if a text was just punctuation). 
    # This drops those empty rows so they don't crash the model.
    df = df.dropna(subset=['cleaned_text'])
    
    # X is our input (the text), y is our output (0 for ham, 1 for spam)
    X = df['cleaned_text']
    y = df['label']
    
    print("2. Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("3. Vectorizing text (converting words to math)...")
    # TF-IDF looks at how often a word appears in a text vs how often it appears overall
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("4. Training the Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    print("5. Evaluating the model...")
    predictions = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    print(f"\n--- MODEL SCORES ---")
    print(f"Accuracy:  {accuracy * 100:.2f}% (How many it got right overall)")
    print(f"Precision: {precision * 100:.2f}% (When it guessed spam, how often was it actually spam?)")
    
    print("\n6. Saving the brain to files...")
    # 'pickle' saves our trained model and vocabulary so our web app can use them later
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Success! model.pkl and vectorizer.pkl are saved.")

if __name__ == "__main__":
    main()