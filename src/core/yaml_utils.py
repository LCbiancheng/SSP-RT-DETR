"""by lyuwenyu
"""

import os
import yaml 
import inspect
import importlib

__all__ = ['GLOBAL_CONFIG', 'register', 'create', 'load_config', 'merge_config', 'merge_dict']


GLOBAL_CONFIG = dict()
INCLUDE_KEY = '__include__'


def _clone_cfg_value(value):
    if isinstance(value, dict):
        return {k: _clone_cfg_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_cfg_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_cfg_value(v) for v in value)
    return value


def register(cls: type):
    '''
    Args:
        cls (type): Module class to be registered.
    '''
    if cls.__name__ in GLOBAL_CONFIG:
        raise ValueError('{} already registered'.format(cls.__name__))

    if inspect.isfunction(cls):
        GLOBAL_CONFIG[cls.__name__] = cls
    
    elif inspect.isclass(cls):
        GLOBAL_CONFIG[cls.__name__] = extract_schema(cls)

    else:
        raise ValueError(f'register {cls}')

    return cls 


def extract_schema(cls: type):
    '''
    Args:
        cls (type),
    Return:
        Dict, 
    '''
    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']
    num_defualts = len(argspec.defaults) if argspec.defaults is not None else 0
    num_requires = len(arg_names) - num_defualts

    schame = dict()
    schame['_name'] = cls.__name__
    schame['_pymodule'] = importlib.import_module(cls.__module__)
    schame['_inject'] = getattr(cls, '__inject__', [])
    schame['_share'] = getattr(cls, '__share__', [])

    for i, name in enumerate(arg_names):
        if name in schame['_share']:
            assert i >= num_requires, 'share config must have default value.'
            value = argspec.defaults[i - num_requires]
        
        elif i >= num_requires:
            value = argspec.defaults[i - num_requires]

        else:
            value = None 

        schame[name] = value
        
    return schame



def create(type_or_name, **kwargs):
    '''
    '''
    assert type(type_or_name) in (type, str), 'create should be class or name.'

    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

    if name in GLOBAL_CONFIG:
        if hasattr(GLOBAL_CONFIG[name], '__dict__'):
            return GLOBAL_CONFIG[name]
    else:
        raise ValueError('The module {} is not registered'.format(name))

    cfg = GLOBAL_CONFIG[name]

    if isinstance(cfg, dict) and 'type' in cfg:
        alias_cfg = _clone_cfg_value(cfg)
        alias_cfg.update(kwargs)
        target_name = str(alias_cfg.pop('type'))
        return create(target_name, **alias_cfg)


    cls = getattr(cfg['_pymodule'], name)
    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']
    
    cls_kwargs = {}
    cls_kwargs.update(_clone_cfg_value(cfg))
    cls_kwargs.update(kwargs)
    
    # shared var
    for k in cfg['_share']:
        if k in kwargs:
            continue
        if k in GLOBAL_CONFIG:
            cls_kwargs[k] = GLOBAL_CONFIG[k]
        else:
            cls_kwargs[k] = cls_kwargs.get(k, cfg[k])

    # inject
    for k in cfg['_inject']:
        if k not in cls_kwargs:
            continue

        inject_value = cls_kwargs[k]

        if inject_value is None:
            continue

        if isinstance(inject_value, str):
            if inject_value not in GLOBAL_CONFIG:
                raise ValueError(f'Missing inject config of {inject_value}.')
            cls_kwargs[k] = create(inject_value)

        elif isinstance(inject_value, dict):
            if 'type' not in inject_value.keys():
                raise ValueError(f'Missing inject for `type` style.')

            inject_cfg = _clone_cfg_value(inject_value)
            _type = str(inject_cfg.pop('type'))
            if _type not in GLOBAL_CONFIG:
                raise ValueError(f'Missing {_type} in inspect stage.')

            cls_kwargs[k] = create(_type, **inject_cfg)

        else:
            # Already-instantiated objects can be forwarded as-is.
            cls_kwargs[k] = inject_value


    cls_kwargs = {n: cls_kwargs[n] for n in arg_names}

    return cls(**cls_kwargs)



def load_config(file_path, cfg=None):
    '''load config
    '''
    if cfg is None:
        cfg = {}

    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"

    with open(file_path, encoding='utf-8') as f:
        file_cfg = yaml.safe_load(f)
        if file_cfg is None:
            return {}

    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith('~'):
                base_yaml = os.path.expanduser(base_yaml)

            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            base_cfg = load_config(base_yaml)
            merge_config(base_cfg, cfg)

    return merge_config(file_cfg, cfg)



def merge_dict(dct, another_dct):
    '''merge another_dct into dct
    '''
    for k in another_dct:
        if (k in dct and isinstance(dct[k], dict) and isinstance(another_dct[k], dict)):
            merge_dict(dct[k], another_dct[k])
        else:
            dct[k] = another_dct[k]

    return dct



def merge_config(config, another_cfg=None):
    """
    Merge config into global config or another_cfg.

    Args:
        config (dict): Config to be merged.

    Returns: global config
    """
    global GLOBAL_CONFIG
    dct = GLOBAL_CONFIG if another_cfg is None else another_cfg
    
    return merge_dict(dct, config)
