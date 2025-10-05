import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib
#Load data 
path_spam = r'C:\Users\prash\AI_Training\Projects\Spam_detector_project\Spam_detector_ML\spam.csv'
df = pd.read_csv(path_spam, encoding='latin-1', usecols=[0,1], names=['label', 'text'], header=0)
X = df['text']
Y = df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
#Create a professional pipeline
# This bundles the vectorizer and the classifier into one object.
pipeline = Pipeline([
    ('vectorizer',CountVectorizer(lowercase=True, token_pattern=r'\b[a-z]*\b')),
    ('classifier', MultinomialNB() )
])
#Training entire pipeline 
pipeline.fit(X_train, Y_train)
print("Model training complete.")
#Evaluating pipeline
accuracy = pipeline.score(X_test,Y_test)
print(f"Model accuracy on test data: {accuracy:.2%}")
# 6. Save the trained pipeline to a file
joblib.dump(pipeline, 'spam_model.pkl')
print("Model pipeline saved to 'spam_model.pkl'")


