from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from glob import glob
import yaml

files = glob('data/regulation/*.yaml')
texts, metadatas, ids = [], [], []

for file in files[:1]:
    content = yaml.safe_load(open(file, 'r'))
    articles = content['articles']
    regulation = content['regulation']
    country = content['country']

    texts += [item['content'] for item in articles]
    metadatas += [
        {**item, 'regulation': regulation, 'country': country}
        for item in articles
    ]
    ids.append(len(ids) + 1)
    break

for text in texts:
    print(repr(text))

texts = texts[:1]
metadatas = metadatas[:1]
ids = ids[:1]

#  Chroma.from_texts(
    #  texts=texts,
    #  embedding=OpenAIEmbeddings(model='text-embedding-3-large'),
    #  metadatas=metadatas,
    #  ids=[str(i) for i in ids],
    #  persist_directory='./my_db'
#  )
