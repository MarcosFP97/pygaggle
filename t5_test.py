import re
import nltk
import pandas as pd
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

    if i + n < len(lst):
        while i + n < len(lst):
            yield lst[i:i + n]
            i += m

    else:
        yield lst


results = []

for d in os.listdir('./test_documents/documents_hannah'):
    
    print("Directory:", d)

    for filename in os.listdir('./test_documents/documents_hannah/'+d):

        print("Filename:", filename)
        doc = ""
        with open('./test_documents/documents_hannah/'+d+'/'+filename) as doc:
            soup =  BeautifulSoup(doc, "html.parser")
            for script in soup(["script", "style"]): # kill all script and style elements
                script.decompose() 
            doc = soup.get_text()
            doc = re.sub('([.!?()])', r'\1 ', doc)
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in doc.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            doc = ' '.join(chunk for chunk in chunks if chunk)
            doc = doc.replace('\\n', ' ').replace('\\r', '').replace('\\t', '').replace('\\', '')

            doc = nltk.sent_tokenize(doc)
            aux = window(doc, 2, 1)
            passages = []
            for chunk in aux:
                
                passages.append(' '.join(chunk))

            model_name = 'castorini/monot5-base-msmarco'
            tokenizer_name = 't5-base'
            reranker =  MonoT5(model_name, tokenizer_name)

            if d=="Topic 7":
                query = Query('Can 5G antennas cause COVID-19?')
            
            elif d=="Topic 15":
                query = Query('Can social distancing prevent COVID-19?')
            
            elif d=="Topic 22":
                query = Query('Can Tamiflu help COVID-19?')

            texts = [Text(p, None, 0) for p in passages] # Note, pyserini scores don't matter since T5 will ignore them.

            reranked = reranker.rerank(query, texts)
            reranked.sort(key=lambda x: x.score, reverse=True)
            print(reranked[0].text)
            dd = {"docId": filename, "snippet": reranked[0].text}
            results.append(dd)

    print("====================================================================")

df = pd.DataFrame(results)
df.set_index('docId', inplace=True)
df.to_csv('snippets.csv')