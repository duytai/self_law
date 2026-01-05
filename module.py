from langchain_redis import RedisCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
import random, os

random.seed(42)

redis_url = 'redis://localhost:6379'
redis_cache = RedisCache(redis_url=redis_url)
set_llm_cache(redis_cache)

gpt4 = ChatOpenAI(
    model='gpt-4.1-mini',
    temperature=0.0,
    max_tokens=2048,
    top_p=1.0,
    api_key=os.environ.get('OPENAI_API_KEY', '')
)

qwen3 = ChatOpenAI(
    model='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    #  model='Qwen/Qwen3-8B',
    temperature=0.0,
    max_tokens=1024,
    top_p=1.0,
    base_url='http://localhost:8000/v1',
    api_key=os.environ.get('OPENROUTER_API_KEY', ''),
    extra_body={
        'chat_template_kwargs': {
            'enable_thinking': False,
        }
    }
)

guard = ChatOpenAI(
    model='Qwen/Qwen3Guard-Gen-4B',
    #  model='meta-llama/Llama-Guard-3-8B',
    #  model='Alibaba-AAIG/Octopus-SEval-14B',
    temperature=0.0,
    max_tokens=20,
    top_p=1.0,
    base_url='http://localhost:8001/v1',
    api_key=os.environ.get('OPENROUTER_API_KEY', ''),
    extra_body={
        'chat_template_kwargs': {
            'enable_thinking': False,
        }
    }
)
