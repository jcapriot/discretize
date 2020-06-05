from pydantic import BaseModel, ValidationError
from pydantic.main import ModelMetaclass
from pydantic.typing import AnyCallable, Optional
from .typing import _serialize_numpy

import numpy as np
from typing import Any, Dict, Union
import re

from .listeners import inherit_listeners, extract_listeners


class ModelWithListenerMetaclass(ModelMetaclass):
    """This metaclass adds a __listeners__ property to the DiscretizeBase
    """
    def __new__(mcs, name, bases, namespace, **kwargs):
        listeners = {}
        for base in reversed(bases):
            if issubclass(base, BaseDiscretize) and base != BaseDiscretize:
                listeners = inherit_listeners(base.__listeners__, listeners)
        listeners = inherit_listeners(extract_listeners(namespace), listeners)

        new_namespace = {
            '__listeners__': listeners,
            **{n: v for n, v in namespace.items()}
        }
        return super().__new__(mcs, name, bases, new_namespace, **kwargs)


class BaseDiscretize(BaseModel, metaclass=ModelWithListenerMetaclass):
    __listeners__: Dict[str, AnyCallable] = {}

    # There are many Config options, we can set it here, then inherit from this class.
    class Config:
        json_encoders = {
            np.ndarray: _serialize_numpy
        }
        validate_assignment = True
        arbitrary_types_allowed = True
        extra = 'allow'

    def __eq__(self, other: Any) -> bool:

        # This is so instances that have numpy arrays can be tested for equivalence
        # This is a recursive function for comparing dictionaries!
        def dict_comparer(dict1, dict2):
            # first check they have all the same keys
            if dict1.keys() != dict2.keys():
                return False
            for key in dict1:
                item1, item2 = dict1[key], dict2[key]
                if isinstance(item1, dict):
                    # Recurse!
                    dict_comparer(item1, item2)
                elif isinstance(item1, np.ndarray):
                    if not np.array_equal(item1, item2):
                        return False
                elif item1 != item2:
                    return False
            return True

        if isinstance(other, BaseModel):
            if self.__class__ != other.__class__:
                return False
            other = other.dict()
        if isinstance(other, dict):
            return dict_comparer(self.dict(), other)
        return self.dict() == other

    def dict(self, *args, **kwargs):
        dic = super().dict(*args, **kwargs)

        # get fully qualified name of class
        dic['__module__'] = self.__class__.__module__
        dic['__class__'] = self.__class__.__name__
        return dic

    def _calculate_keys(
        self,
        include: Optional[Union['AbstractSetIntStr', 'DictIntStrAny']],
        exclude: Optional[Union['AbstractSetIntStr', 'DictIntStrAny']],
        exclude_unset: bool,
        update: Optional['DictStrAny'] = None,
    ) -> Optional['SetStr']:
        """This will cause private values that start with _ to not be included in dict, copy, etc"""
        __fields_set__ = getattr(self, '__fields_set__', None)
        if __fields_set__:
            if exclude is None:
                exclude = set()
            for field in __fields_set__:
                if re.match('_.*?', field):
                    exclude.add(field)
        return super()._calculate_keys(include, exclude, exclude_unset, update=update)

    def __setattr__(self, name, value):
        """adds the listener function to get called AFTER value has been set and validated"""
        super().__setattr__(name, value)
        if name in self.__listeners__:
            for listener in self.__listeners__[name]:
                listener.func(self, value)

    def _easy_validate(self, name, value):
        """A simple method to call if you specifically need to validate a value"""
        field = self.__fields__.get(name, None)
        if field:
            value, error = field.validate(value, self.dict(exclude={name}), loc=name, cls=self.__class__)
            if error:
                raise ValidationError([error], self.__class__)
        return value
