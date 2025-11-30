from langchain_redis import RedisCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
import random

random.seed(42)

redis_url = 'redis://localhost:6379'
redis_cache = RedisCache(redis_url=redis_url)
set_llm_cache(redis_cache)
llm = ChatOpenAI(
    model='gpt-4.1-mini',
    temperature=0.0,
    max_tokens=2048,
    top_p=1.0,
)