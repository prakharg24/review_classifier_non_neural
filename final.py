import json
import codecs
import random
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from nltk.tokenize import RegexpTokenizer
import pickle
import sys

def getclassmore(a):
    if a>3: return a-2
    elif a<3: return 0
    else: return 1

def getclass(a):
    if a>3:
        return 2
    elif a<3:
        return 0
    else:
        return 1

def allupper(word):
    for c in word:
        if not(c.isupper()):
            return False
    return True
    
def cleandoc(doc):
    global imp
    unclean = doc.split()
    words = []
    for word in unclean:
        if len(word)>2:
            words.append(word)
        if word in imp:
            for i in range(0, 3):
                words.append(word)
    lng = len(words)
    for i in range(0, lng):
        word = words[i]
        if allupper(word):
            words.append(word)
        if word=="not":
                for j in range(1, 5):
                    if(i+j<lng):
                        words[i+j]="NOT_" + words[i+j]
    lower_words = [word.lower() for word in words]
    return ' '.join(lower_words)

print("Reading side files")
imp = set()

file = open('adjectives.txt', 'r')
for adj_en in file.readlines():
    imp.add(adj_en.split()[0])

file = open('adverbs.txt', 'r')
for adj_en in file.readlines():
    imp.add(adj_en.split()[0])

file = open('verbs.txt', 'r')
for adj_en in file.readlines():
    imp.add(adj_en.split()[0])


print("Reading test json file")
test_data = []
test_it = 0
with codecs.open(sys.argv[1],'rU','utf-8') as f:
	for line in f:
		test_it = test_it + 1
		test_data.append(json.loads(line))

print("Cleaning test sentences")

test_sentences = []
end = test_it
i = 0
while(i<end):
    sent = test_data[i]['reviewText']
    temp = ""
    for j in range(0, 3):
        temp = temp + test_data[i]['summary']
    sent = sent + temp
    test_sentences.append(cleandoc(sent))
    i = i+1

with open('vect_uni.pkl', 'rb') as f:
    vect_uni = pickle.load(f)

with open('vect_bi.pkl', 'rb') as f:
    vect_bi = pickle.load(f)

print("Making Test matrix - Unigrams")
test_matrix_uni = vect_uni.transform(test_sentences)

print("Making Test matrix - Unigrams")
test_matrix_bi = vect_bi.transform(test_sentences)

test_matrix = hstack((test_matrix_uni, test_matrix_bi))

print("Predicting")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
y_pred = model.predict(test_matrix)
y_pred_class = []
for ele in y_pred:
    if(ele==3):
        y_pred_class.append(2)
    else:
        y_pred_class.append(ele)


file = open(sys.argv[2],'w')
for ele in y_pred_class:
    if(ele==0):
        file.write("1\n")
    elif(ele==1):
        file.write("3\n")
    elif(ele==2):
        file.write("5\n")