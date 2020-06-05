import numpy as np
import warnings


class TypedArray():
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def __modify_schema__(cls, field_schema):
        if issubclass(cls.npdtype.type, np.integer):
            base_name = 'integer'
        elif issubclass(cls.npdtype.type, np.bool):
            base_name = 'bool'
        else:
            base_name = 'number'
        field_schema.update(
            type='array',
            shape={'type': 'array'},
            np_type={'type': 'string'},
            f_order={'type': 'bool'},
            items={'type': base_name},
        )
        field_schema['shape'].update(items={'integer'})

    @classmethod
    def validate_type(cls, val):
        if isinstance(val, dict):
            val = _deserialize_numpy_dict(val)
        val = np.asarray(val, dtype=cls.npdtype)
        if cls.shape is None:
            return val
        if len(cls.shape) != len(val.shape):
            raise ValueError(f'Array incorrect dimension, expected dimension: {len(cls.shape)}, '
                             f'input dimension : {len(val.shape)}'
            )
        for n1, n2 in zip(cls.shape, val.shape):
            if n1 > 0 and n1 != n2:
                raise ValueError(f'Array incorrect shape, expected shape: {cls.shape}, '
                             f'input shape : {val.shape}'
                )
        return val


class UnitaryTypedArray(TypedArray):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_unit

    @classmethod
    def validate_unit(cls, val):
        val = cls.validate_type(val)
        arr_norm = np.linalg.norm(val)
        if arr_norm == 0:
            raise ValueError('Input vector has 0 length')
        if arr_norm < 1E-16:
            warnings.warn('Input vector length close to 0')
        return val/arr_norm


class ArrayMeta(type):
    def __getitem__(self, t):
        if isinstance(t, tuple):
            shape = t[1]
            if isinstance(shape, int):
                shape = (shape, )
            t = t[0]
        else:
            shape = None
        if issubclass(np.dtype(t).type, (np.complex256, np.float128)):
            raise TypeError(f'{t} is unsorported')
        return type('Array', (TypedArray,), {'npdtype': np.dtype(t), 'shape': shape})


class UnitaryArrayMeta(type):
    def __getitem__(self, t):
        if isinstance(t, tuple):
            shape = t[1]
            if isinstance(shape, int):
                shape = (shape, )
            t = t[0]
        else:
            shape = None
        if issubclass(np.dtype(t).type, (np.complex256, np.float128, np.bool)):
            raise TypeError(f'{t} is unsorported')
        return type('Array', (UnitaryTypedArray,), {'npdtype': np.dtype(t), 'shape':shape})


class Array(np.ndarray, metaclass=ArrayMeta):
    pass


class UnitaryArray(np.ndarray, metaclass=UnitaryArrayMeta):
    pass


def _serialize_numpy(x):
    shape = x.shape
    f_order = x.flags['F_CONTIGUOUS']
    x = x.flatten(order='A')
    np_type = x.dtype
    if issubclass(np_type.type, np.complexfloating):
        if x.itemsize == 8:
            x = x.view(dtype=np.float32)
        elif x.itemsize == 16:
            x = x.view(dtype=np.float64)
    return {'shape': shape, 'np_type': str(np_type), 'f_order': f_order, 'items': x.tolist()}


def _deserialize_numpy_dict(dic):
    dtype = dic['np_type']
    if 'complex' in dtype:
        item_dtype = f'float{int(dtype[7:])//2}'
    else:
        item_dtype = dtype

    out = np.array(dic['items'], dtype=item_dtype)
    if 'complex' in dtype:
        out = out.view(dtype=dtype)
    if dic['f_order']:
        out = out.reshape(dic['shape'], order='F')
    else:
        out = out.reshape(dic['shape'])
    return out
