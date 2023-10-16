import csv
import string
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

def clean_term(text):
    text = text.lower()
    return "".join(char for char in text
                   if char not in string.punctuation)


#小文字化+句読点削除
def toLower(text):
    nltk_tokens = nltk.word_tokenize(text)
    result = ''
    for w in nltk_tokens:
            text = clean_term(w)
            if not text.isdigit():
                result = result + ' ' + text
    return result

#小文字化+句読点削除+ステム+レンマ
def stemmerLemmatizer(text):
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    nltk_tokens = nltk.word_tokenize(text)
    result = ''
    for w in nltk_tokens:
        text = clean_term(w)
        if not text.isdigit():
            result = result + ' ' + stemmer.stem(wordnet_lemmatizer.lemmatize(text))
    return result

#ストップワード削除
def stop_wordsAll(text):
    nltk_tokens = nltk.word_tokenize(text)
    result = ''
    for w in nltk_tokens:
        if w not in stop_words:
            text = clean_term(w)
            if not text.isdigit():
                result = result + ' ' + text
    return result

#ストップワード削除
stop_words_2 = ['the', 'a', 'this', 'that']
def stop_wordsSmall(text):
    nltk_tokens = nltk.word_tokenize(text)
    result = ''
    for w in nltk_tokens:
        if w not in stop_words_2:
            text = clean_term(w)
            if not text.isdigit():
                result = result + ' ' + text
    return result

#小文字化+句読点削除+ステム化+レンマ化+ストップワード削除
def standardize(text):
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    nltk_tokens = nltk.word_tokenize(text)
    result = ''
    for w in nltk_tokens:
        if w not in stop_words:
            text = clean_term(w)
            if not text.isdigit():
                result = result + ' ' + stemmer.stem(wordnet_lemmatizer.lemmatize(text))
    return result

# satd_comment = '// JUnit 4 wraps solo tests this way. We can extract // the original test name with a little hack.'

# ans = standardize(satd_comment)