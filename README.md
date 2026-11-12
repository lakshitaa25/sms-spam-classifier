# 🛡️ SMS Spam Classifier

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to accurately classify SMS text messages as **Spam** or **Ham (Legitimate)**.

---

## 📁 Project Architecture

```
sms-spam-classifier/
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── data/
│   └── sms_spam_sample.csv     # Labeled SMS dataset (Ham vs Spam)
├── models/
│   ├── spam_classifier_model.joblib # Saved trained ML model (auto-generated)
│   ├── tfidf_vectorizer.joblib      # Saved TF-IDF vectorizer (auto-generated)
│   └── metadata.json                # Model performance metrics & dataset stats
└── src/
    ├── preprocess.py           # NLP text cleaning, tokenization, stemming
    ├── train.py                # Model training, comparison & evaluation pipeline
    └── predict.py              # CLI and programmatic inference interface
```

---

## ⚡ Quickstart

### 1. Navigate to the project directory

```powershell
cd c:\Users\Hp\gesture.ai\sms-spam-classifier
```

### 2. Set up a virtual environment (Recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 🚀 Running the Web Application

Launch the interactive Streamlit web dashboard:

```powershell
streamlit run app.py
```

### Features of the Web App:
- **Instant Spam Detection**: Type or paste any SMS message to evaluate.
- **One-Click Presets**: Pre-loaded examples of lottery spam, phishing alerts, OTP codes, and regular chats.
- **Confidence Scoring**: Visual probability distribution (Spam % vs Legitimate %).
- **Suspicious Keyword Highlighting**: Identifies flagged terms (e.g., *winner, claim, http_token, cash*).
- **Model Dashboard**: Live view of precision, recall, accuracy, and dataset balance.
- **One-Click Retrain**: Retrain the model directly from the UI.

---

## 💻 CLI & Programmatic Usage

### Command Line Interface

Classify any message directly from your terminal:

```powershell
python src/predict.py --message "URGENT! You have won £1,000 cash. Call 09061701461 to claim now."
```

Or for a legitimate message:

```powershell
python src/predict.py --message "Hey, are we still meeting for lunch today at 12:30?"
```

### Python API

Integrate the classifier into your own Python code:

```python
from src.predict import SMSClassifier

classifier = SMSClassifier()
result = classifier.predict("Congratulations! You won a $1,000 gift card. Click here to claim.")

print(result["label"])             # 'SPAM'
print(result["confidence"])        # 0.98
print(result["spam_probability"])  # 0.98
print(result["flagged_keywords"])  # ['congratulations', 'claim', 'card', 'gift']
```

---

## 🧠 Model Training & NLP Pipeline

Run the training pipeline manually:

```powershell
python src/train.py
```

### How the Pipeline Works:
1. **Data Cleaning & Normalization**:
   - URL normalization (`http_token`)
   - Phone & short-code normalization (`phone_token`)
   - Currency symbol normalization (`money_token`)
   - Removal of punctuation and non-alphanumeric noise
   - Stopword filtering and Porter Stemming
2. **Feature Extraction**:
   - Sublinear TF-IDF (Term Frequency-Inverse Document Frequency)
   - Unigram + Bigram feature capture (`ngram_range=(1, 2)`)
3. **Model Selection**:
   - Evaluates **Multinomial Naive Bayes**, **Logistic Regression**, and **Calibrated Linear SVM**.
   - Prioritizes **Precision** on the Spam class to prevent false positives (so legitimate personal messages are never wrongly blocked).
4. **Export**:
   - Automatically exports model artifacts and evaluation metrics to `models/`.

### Using the Full Kaggle / UCI SMS Dataset:

#### Method 1: Automatic 1-Step Downloader (Easiest)
Run the built-in downloader script to automatically fetch and configure the full 5,574-message UCI SMS Spam dataset:
```powershell
python src/download_data.py
```
This automatically saves the processed data as `data/spam.csv`.

#### Method 2: Manual Download from Kaggle
1. Download `spam.csv` from [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
2. Save it directly to `sms-spam-classifier/data/spam.csv`.

#### Train on the Full Dataset:
Once the dataset is in place, run:
```powershell
python src/train.py
```
The pipeline automatically prioritizes `data/spam.csv` over the sample dataset.
