import sklearn
from typing import Type, Dict, AnyStr, List, Union, Any, TypeVar, NewType, TypeAlias

HPTParamsValList = List[float] | List[str]
HPTParamsDict = Dict[AnyStr, HPTParamsValList]
ParamsDict = Dict[AnyStr, int | HPTParamsDict]

# https://stackoverflow.com/questions/58755948/what-is-the-difference-between-typevar-and-newtype
# https://stackoverflow.com/questions/33045222/how-do-you-alias-a-type-in-python
# https://docs.python.org/3.12/whatsnew/3.12.html#pep-695-type-parameter-syntax
ColumnTransformer: TypeAlias = sklearn.compose._column_transformer.ColumnTransformer
