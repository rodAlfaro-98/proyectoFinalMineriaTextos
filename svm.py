import nltk
import pandas as pd
from nltk.stem import *
from nltk import stem
from sklearn.model_selection import train_test_split
#nltk.download('punkt') # For Stemming
#nltk.download('wordnet') # For Lemmatization
#nltk.download('stopwords') # For Stopword Removal
#nltk.download('omw-1.4')
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = stem.SnowballStemmer('english')
stopwords.add('user')
stopwords.add('ca')
#stopwords.add('n\'t')
#print(stopwords)

data = pd.read_csv('./datasets/train.csv')
data = data[['label','tweet']]

def review_messages(msg):
    # converting messages to lowercase
    msg = msg.lower()
    # removing stopwords
    msg = [word for word in msg.split() if word not in stopwords]
    # using a stemmer
    msg = " ".join([stemmer.stem(word) for word in msg])
    return msg

data['tweet'] = data['tweet'].apply(review_messages)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['label'], test_size = 0.1, random_state = 1)
# training the vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

from sklearn import svm
svm = svm.SVC(C=1000)
svm.fit(X_train, y_train)



from sklearn.metrics import confusion_matrix
X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test)
#print(confusion_matrix(y_test, y_pred))
