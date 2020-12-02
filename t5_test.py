import re
import nltk
from bs4 import BeautifulSoup 
import os 
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

def window(iterable, n = 6, m = 3):
    if m == 0: # otherwise infinte loop
        raise ValueError("Parameter 'm' can't be 0")
    lst = list(iterable)
    i = 0
    while i + n < len(lst):
        yield lst[i:i + n]
        i += m

doc = ""
with open('doc1_ibupr.txt') as f:
    doc = f.readlines()[0]

soup =  BeautifulSoup(doc, "html.parser")
doc = soup.get_text()
doc = doc.replace('\\n', ' ').replace('\\r', '').replace('\\t', '').replace('\\', '')

doc = nltk.sent_tokenize(doc)
aux = window(doc)
passages = []
for chunk in aux:
    passages.append(' '.join(chunk))

model_name = 'castorini/monot5-base-msmarco'
tokenizer_name = 't5-base'
reranker =  MonoT5(model_name, tokenizer_name)

query = Query('Ibuprofen worsens COVID-19')

texts = [Text(p, None, 0) for p in passages] # Note, pyserini scores don't matter since T5 will ignore them.

reranked = reranker.rerank(query, texts)
reranked.sort(key=lambda x: x.score, reverse=True)

for i in range(10):
    print(f'{reranked[i].score} {reranked[i].text}')
    print("=======================================")
    print()
