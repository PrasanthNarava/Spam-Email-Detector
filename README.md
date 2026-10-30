# Spam Message Detector

Simple Streamlit web application that classifies messages as **Spam** or **Ham** using a Multinomial Naive Bayes model.

Trained on the classic SMS Spam Collection dataset and packaged with a ready-to-use model file.

---

## Features

- Real-time spam / ham classification
- Confidence score (probability) display
- Clean Streamlit interface
- Pre-trained model included (`spam_model.pkl`)
- Easy to retrain with `train.py`

---

## Tech Stack

- **Python**
- **scikit-learn** (MultinomialNB + CountVectorizer)
- **Streamlit** (web UI)
- **joblib** (model persistence)

---

## Project Structure

```
Spam-Email-Detector/
├── app.py              # Streamlit application
├── train.py            # Model training script
├── spam_model.pkl      # Pre-trained pipeline
├── spam.csv            # Training dataset
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install streamlit scikit-learn joblib pandas
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open in your browser. Enter a message and click **Check Message** to see the classification and confidence score.

### 3. (Optional) Retrain the model

```bash
python train.py
```

This will regenerate `spam_model.pkl` from `spam.csv`.

---

## How It Works

1. Text is vectorized using `CountVectorizer`
2. Multinomial Naive Bayes predicts the class
3. Prediction probabilities are returned as a confidence score
4. Streamlit displays the result with clear visual feedback

---

## Example

**Input**  
`Congratulations! You've won a $1000 prize. Click here to claim.`

**Output**  
🚨 This message is classified as **SPAM** (high confidence)

---

**Author**: Prasanth Narava  
**Focus**: Practical ML applications • Lightweight classification tools  
**Last Updated**: October 2026
