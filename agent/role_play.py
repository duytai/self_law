from typing import Dict
from omegaconf import OmegaConf
from litellm.caching.caching import Cache
from pydantic import BaseModel, Field
from utils import make_dataset 
from jinja2 import Template
import litellm

litellm.cache = Cache(type='disk')
litellm.cache_only = True

class ConversationMessage(BaseModel):
    name: str = Field(description="...")
    content: str = Field(description='...')

class Conversation(BaseModel):
    message: List[ConversationMessage] = Field(description="...")

def resolve(conf, **overrides) -> Dict:
    cfg = OmegaConf.to_yaml(conf)
    cfg = Template(cfg).render(overrides)
    cfg = OmegaConf.create(cfg)
    return OmegaConf.to_container(cfg)

def app():
    conf = OmegaConf.load('config/role_play.yaml')
    dataset = make_dataset(conf.dataset)
    scenario = dataset[0]

    question = resolve(conf.question, scenario=scenario)
    response = litellm.completion(**question)

    print('----')
    print(response.choices[-1].message.content)

if __name__ == '__main__':
    app()
