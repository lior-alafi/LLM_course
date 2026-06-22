from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import configparser, os
DEFAULT_CONFIG_PATH = Path.home() / 'doit.cfg'
@dataclass(frozen=True)
class ModelConfig:
    provider: str; model: str;api_key:str|None=None; api_base: str|None=None; api_key_env: str|None=None
@dataclass(frozen=True)
class AgentConfig:
    shell: str='/bin/bash'; secure: bool=False; max_clarification_rounds:int=2
@dataclass(frozen=True)
class StateConfig:
    backend: str='file'; redis_url: str|None=None
@dataclass(frozen=True)
class MemoryConfig:
    backend: str='file'; redis_url: str|None=None
@dataclass(frozen=True)
class UserAwarenessConfig:
    enabled: bool=True; shell_history_limit:int=20
@dataclass(frozen=True)
class ContextConfig:
    summaries_enabled: bool=True; summary_threshold:int=20; summary_recent_keep:int=8
@dataclass(frozen=True)
class DoitConfig:
    model: ModelConfig; agent: AgentConfig; state: StateConfig; memory: MemoryConfig; user_awareness: UserAwarenessConfig; context: ContextConfig

def _parse_bool_env(value: str|None) -> bool|None:
    if value is None: return None
    v=value.strip().lower()
    if v in {'1','true','yes','y','on'}: return True
    if v in {'0','false','no','n','off'}: return False
    raise ValueError(f'Invalid boolean value for DOIT_SECURE: {value}')
class ConfigLoader:
    def __init__(self, path:Path|None=None): self.path=path or DEFAULT_CONFIG_PATH
    def load(self)->DoitConfig:
        p=configparser.ConfigParser()
        if self.path.exists(): p.read(self.path)
        provider=p.get('model','provider',fallback='ollama'); model=p.get('model','model',fallback='ollama/gemma3:4b')
        api_key=p.get('model','api_key',fallback=None); api_base=p.get('model','api_base',fallback=None); api_key_env=p.get('model','api_key_env',fallback=None)
        if api_key is None and api_key_env:
            api_key = os.getenv(api_key_env)
        # if api_key is None:
        #      raise RuntimeError("No API key configured")
        shell=p.get('agent','shell',fallback='/bin/bash')
        secure_cfg=p.getboolean('agent','secure',fallback=False); secure_env=_parse_bool_env(os.getenv('DOIT_SECURE'))
        secure=secure_env if secure_env is not None else secure_cfg
        max_clar=p.getint('agent','max_clarification_rounds',fallback=2)
        state_backend=p.get('state','backend',fallback='file'); state_redis=p.get('state','redis_url',fallback=None)
        mem_backend=p.get('memory','backend',fallback='file'); mem_redis=p.get('memory','redis_url',fallback=None)
        ua_enabled=p.getboolean('user_awareness','enabled',fallback=True); shell_hist_limit=p.getint('user_awareness','shell_history_limit',fallback=20)
        summ_enabled=p.getboolean('context','summaries_enabled',fallback=True); summ_threshold=p.getint('context','summary_threshold',fallback=20); summ_keep=p.getint('context','summary_recent_keep',fallback=8)
        return DoitConfig(ModelConfig(provider,model,api_key,api_base,api_key_env), AgentConfig(shell,secure,max_clar), StateConfig(state_backend,state_redis), MemoryConfig(mem_backend,mem_redis), UserAwarenessConfig(ua_enabled,shell_hist_limit), ContextConfig(summ_enabled,summ_threshold,summ_keep))
