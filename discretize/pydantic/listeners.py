from pydantic.utils import in_ipython
from pydantic.errors import ConfigError
from pydantic.typing import AnyCallable
from typing import Any, Dict, List

# these are largely modeled after the validators of pydantic
_FUNCS = set()
LISTENER_CONFIG_KEY = '__listener_config__'


class Listener:
    __slots__ = 'func'

    def __init__(
        self,
        func: AnyCallable,
    ):
        self.func = func


def listener(field: str, allow_reuse: bool = False):
    def dec(f: AnyCallable):
        f = _prepare_listener(f, allow_reuse)
        setattr(f, LISTENER_CONFIG_KEY, (field, Listener(func=f)))
        return f
    return dec


def _prepare_listener(function: AnyCallable, allow_reuse: bool):
    # do not make it a class method
    if not in_ipython() and not allow_reuse:
        ref = function.__module__ + '.' + function.__qualname__
        if ref in _FUNCS:
            raise ConfigError(f'duplicate validator function "{ref}"; if this is intended, set `allow_reuse=True`')
        _FUNCS.add(ref)
    return function


def inherit_listeners(base_listeners, listeners):
    for field, field_listener in base_listeners.items():
        if field not in listeners:
            listeners[field] = []
        listeners[field] += field_listener
    return listeners


def extract_listeners(namespace: Dict[str, Any]) -> Dict[str, List[Listener]]:
    listeners: Dict[str, List[Listener]] = {}
    for var_name, value in namespace.items():
        listener_config = getattr(value, LISTENER_CONFIG_KEY, None)
        if listener_config:
            fields, v = listener_config
            for field in fields:
                if field in listeners:
                    listeners[field].append(v)
                else:
                    listeners[field] = [v]
    return listeners
