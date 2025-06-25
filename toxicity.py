from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.ensemble  import RandomForestClassifier
from xgboost import XGBClassifier
import re
import string
import numpy as np
import pandas as pd
from sentence_transformers import util


df = pd.read_csv(r"C:\Users\nakul\Downloads\toxicity_en.csv")


hate_sentences = df[df['is_toxic'].str.lower() == 'toxic']['text'].tolist()
not_hate_sentences = df[df['is_toxic'].str.lower() == 'not toxic']['text'].tolist()





from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')

hate_embeddings = model.encode(hate_sentences)
not_hate_embeddings = model.encode(not_hate_sentences)


X = model.encode(df['text'].tolist(), show_progress_bar=True)
y = df['is_toxic']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression()
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))



	


from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier


ensemble = VotingClassifier(estimators=[
    ('lr', LogisticRegression()),
    ('xgb2', XGBClassifier()),
    ('lr3', LogisticRegression()),
    ('xgb', XGBClassifier())
], voting='soft')

ensemble.fit(X_train, y_train)

y_pred2 = ensemble.predict(X_test)
print(classification_report(y_test, y_pred2))




def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)            
    text = re.sub(r'@\w+', '', text)               
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)                
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_text(text):
    clean = clean_text(text)
    embedding = model.encode([clean])

    model_prob = lr.predict_proba(embedding)[0]  
    
    

    hate_similarity = np.mean(util.cos_sim(embedding, hate_embeddings).numpy())
    not_hate_similarity = np.mean(util.cos_sim(embedding, not_hate_embeddings).numpy())

  
    if  hate_similarity>not_hate_similarity:
        adj_hate = model_prob[1] + hate_similarity
        adj_not_hate = model_prob[0] - not_hate_similarity
    else:
        adj_hate = model_prob[1] - not_hate_similarity
        adj_not_hate = model_prob[0] + not_hate_similarity 
   
    

    
    total = adj_hate + adj_not_hate
    adj_hate /= total
    adj_not_hate /= total

    result = "Hate Speech" if adj_hate > adj_not_hate else "Not Hate"  
    return result



tweet = input("Enter a comment to check: ")
result = predict_text(tweet)
print("Prediction:", result)
