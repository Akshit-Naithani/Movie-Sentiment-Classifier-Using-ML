import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Data Preprocessing
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getStemmedReview(review):
    review = review.lower()
    review = review.replace("<br /><br />", " ")
    
    tokens = tokenizer.tokenize(review)
    stopped_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in stopped_tokens]
    
    cleaned_review = " ".join(stemmed_tokens)
    
    return cleaned_review

# Read dataset
dataset = pd.read_csv("C:/Python Files/Python Project College/ML/IMDB Dataset.csv")

# Cleaning the reviews
dataset['cleaned_review'] = dataset['review'].apply(getStemmedReview)

# Feature Extraction
cv = CountVectorizer()
X = cv.fit_transform(dataset['cleaned_review'])
y = (dataset['sentiment'] == 'positive').astype(int)

# Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", format((accuracy*100), ".2f"),'%')

