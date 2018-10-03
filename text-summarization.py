# -*- coding: utf-8 -*-
"""
@author: moham
"""

import bs4 as bs
import urllib.request
import re
import nltk
from nltk.corpus import stopwords
import heapq
source=urllib.request.urlopen('https://en.wikipedia.org/wiki/Deep_learning').read()

soup=bs.BeautifulSoup(source,'lxml')

text=''
for paragraph in soup.find_all('p'):
    text+=paragraph.text


text=re.sub('\[[0-9]*\]',' ',text)
text=re.sub('\s+',' ',text)
clean_text=text.lower()
clean_text=re.sub('\W',' ',clean_text)
clean_text=re.sub('\d',' ',clean_text)
clean_text=re.sub('\s[a-zA-Z0-9]\s',' ',clean_text)
clean_text=re.sub('\s+',' ',clean_text)

sentences=nltk.sent_tokenize(text)
#for removing it from my histogram words
stop=stopwords.words('english')

words=nltk.word_tokenize(clean_text)
# =============================================================================
# new_word=[]
# for word in words:
#     if word in stop:
#         continue
#     else:
#         new_word.append(word)
# =============================================================================

histo={}
for word in nltk.word_tokenize(clean_text):

        if (word in histo.keys()) and (word not in stop):
            histo[word]+=1
        else:
            histo[word]=1


for key in histo.keys():
    histo[key]=histo[key]/max(histo.values())
    
    
#sentence scores 
sent2score={}
for sent in sentences:
    score=0
    for word in nltk.word_tokenize(sent.lower()):
        if word in histo.keys():
            score+=histo[word]
    if len(sent.split(' '))<30:
        if sent not in sent2score.keys():
            sent2score[sent]=score
        else:
            sent2score[sent]+=score

best_sent=heapq.nlargest(5,sent2score,key=sent2score.get)
print(*best_sent)