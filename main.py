import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
from collections import defaultdict, Counter
import numpy as np
import math
import os

#DEPENDENCIES
#nltk.download('stopwords')
#nltk.download('punkt')    

n = int(input("Enter number of documents :"))
corpus = []
for i in range(n):
    corpus.append(input("Enter doc no "+str(i+1)+" :"))
"""corpus = [
    'this is the FIRST!! document',
    'this document is.,,,.! the second document',
    'and this is the third one',
    'is this the last document',
]"""
doc_count = len(corpus)     #Total number of documents

def convert_lower_case(data):
    return np.char.lower(data)

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def preprocessing(doc):
    doc = convert_lower_case(doc)
    doc = remove_punctuation(doc)
    doc = remove_apostrophe(doc)
    doc = remove_stop_words(doc)
    doc = convert_numbers(doc)
    doc = stemming(doc)
    doc = remove_punctuation(doc)
    doc = convert_numbers(doc)
    doc = stemming(doc)
    doc = remove_punctuation(doc)
    doc = remove_stop_words(doc)
    return doc

def get_inverted_index(Findex):
    Iindex = defaultdict(dict)
    subdict = dict()
    for doc_num,index in Findex.items():
        for token in index.keys():
            subdict[doc_num]=index.get(token)
            Iindex[token].update(subdict)
            subdict.clear()
    return Iindex

def documentFrequency(token):
	return len(inverted_index[token])

def termFrequency(token,doc):
	return inverted_index[token].get(doc,0)/sum(forward_index[doc].values())

def tf_idfWeighingAndVectorization(query):
	Scores = defaultdict(float)
	Length = defaultdict(float)
	q_total = len(query)
	q_counter = Counter(query)
	for token in set(query):
		TFq = q_counter[token]/q_total
		DF = documentFrequency(token)
		IDF = 0
		if DF!=0:IDF = math.log10(docid/DF)
		Wq = TFq*IDF
		for d in inverted_index[token].keys():
			TFd = termFrequency(token,d)
			Wd = TFd*IDF
			Scores[d]+=Wq*Wd

	Length = {docno:sum(Score**2 for Score in Scores.values()) for docno in Scores.keys()}
	return Scores,Length

docid = 0
forward_index = defaultdict(dict)
for doc in corpus:

    #PREPROCESS DATA
    doc = preprocessing(doc)

    doc = doc.split()
    docid+=1

    #FORWARD INDEX
    forward_index[docid]=dict(Counter(doc))

#INVERTED INDEX
inverted_index = get_inverted_index(forward_index)

#SEARCH INPUT
print()
search = input("ENTER SEARCH QUERRY :")
search = preprocessing(search)
search = search.split()

TF,magn = tf_idfWeighingAndVectorization(search)
Cosine_Score = {}
for d in TF.keys():
	Cosine_Score[d]=TF[d]/math.sqrt(magn[d])
print("DOCUMENT NUMBER", "| COSINE SCORE")
for docno in range(doc_count):
    print(docno,"\t|\t", Cosine_Score.get(docno+1,0))
    
Cosine_Score = sorted(Cosine_Score.items(), key=lambda item: item[1], reverse=True)
print()
print("MOST RELAVANT DOCUMENTS (descending order of relevancy)")
print(*[str(docno)+" - "+corpus[docno-1] for docno,score in Cosine_Score],sep="\n")
