import nltk
import pandas as pd
import re
from nltk.stem import *
from sklearn.model_selection import train_test_split
#nltk.download('punkt') # For Stemming
#nltk.download('wordnet') # For Lemmatization
#nltk.download('stopwords') # For Stopword Removal
#nltk.download('omw-1.4')

# Remove URLs
def remove_ð(text):
    text = re.sub(r'ð[^\w]*','',text)
    return re.sub(r'ð[^\w]*','',text)

#Eliminación de stopwords para crear un corpus
def text_preprocessing(df,column):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.add('user')
    stopwords.add('ca')
    corpus=[]

    lem = WordNetLemmatizer() # For Lemmatization
    for emotion in df[column]:
        words=[w for w in nltk.tokenize.word_tokenize(emotion) if not w in stopwords] # word_tokenize function tokenizes text on each word by default
        words=[lem.lemmatize(w) for w in words if len(w)>=2]
        words = list(map(lambda w:w.replace('n\'t', 'not'), words))
        corpus.append(words)
    return corpus

def message_preprocessing(message):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.add('user')
    stopwords.add('ca')
    corpus=[]

    lem = WordNetLemmatizer() # For Lemmatization
    words=[w for w in nltk.tokenize.word_tokenize(message) if not w in stopwords] # word_tokenize function tokenizes text on each word by default
    words=[lem.lemmatize(w) for w in words if len(w)>=2]
    words = list(map(lambda w:w.replace('n\'t', 'not'), words))
    corpus.append(words)
    return corpus

def getProbabilities():

    df = pd.read_csv('./datasets/train.csv')

    # Apply this function on our data frame
    df['tweet'] = df['tweet'].apply(remove_ð)
    corpus = text_preprocessing(df,'tweet')
    hate_tweets_indexes = df[df['label'] == 1].index
    no_hate_tweets_indexes = df[df['label'] == 0].index

    hate_corpus = []
    for i in hate_tweets_indexes.values:
        hate_corpus.append(corpus[i])

    no_hate_corpus = []
    for i in no_hate_tweets_indexes.values:
        no_hate_corpus.append(corpus[i])

    total_hate = 0
    for i in hate_corpus:
        total_hate += len(i)

    total_no_hate = 0
    for i in no_hate_corpus:
        total_no_hate += len(i)

    total = total_hate + total_no_hate

    unique_word_in_hate = set([j for i in hate_corpus for j in i])
    unique_word_in_no_hate = set([j for i in no_hate_corpus for j in i])

    hate_aux = [j for i in hate_corpus for j in i]
    no_hate_aux = [j for i in no_hate_corpus for j in i]

    hate_words = {}
    for word in unique_word_in_hate:
        hate_words[word] = hate_aux.count(word)

    print(len(hate_words))

    no_hate_words = {}
    for word in unique_word_in_no_hate:
        no_hate_words[word] = no_hate_aux.count(word)

    print(len(no_hate_words))

    df_hate_words = pd.Series(hate_words)
    df_hate_words.to_csv('./datasets/hate_words_prob.csv')

    df_no_hate_words = pd.Series(no_hate_words)

    df_no_hate_words.to_csv('./datasets/no_hate_words_prob.csv')

if __name__ == "__main__":
    print("Generando archivos de probabilidades de odio y no odio")
    getProbabilities()