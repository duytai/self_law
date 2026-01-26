from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from glob import glob
from tqdm import tqdm
import yaml

files = glob('data/regulation/*.yaml')
texts, metadatas, ids = [], [], []

for file in tqdm(files):
    content = yaml.safe_load(open(file, 'r'))
    articles = content['articles']
    regulation = content['regulation']
    country = content['country']

    texts += [item['content'] for item in articles]
    metadatas += [
        {**item, 'regulation': regulation, 'country': country}
        for item in articles
    ]
ids = list(range(len(texts)))

Chroma.from_texts(
    texts=texts,
    embedding=OpenAIEmbeddings(model='text-embedding-3-large'),
    metadatas=metadatas,
    ids=[str(i) for i in ids],
    persist_directory='./reg_db'
)
