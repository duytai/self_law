from langchain_redis import RedisCache
from langchain_core.globals import set_llm_cache
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.storage.redis import RedisStore
import random

random.seed(42)

redis_url = 'redis://localhost:6379'
redis_cache = RedisCache(redis_url=redis_url)
set_llm_cache(redis_cache)
llm = ChatOpenAI(model='gpt-4.1-mini')

redis_byte_store = RedisStore(redis_url=redis_url)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, redis_byte_store)