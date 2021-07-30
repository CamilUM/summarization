from sklearn.model_selection import ParameterGrid
from functools import partial

def grid(method, train, documents, params):
    summarizers = {}
    pg = ParameterGrid(params)
    for g in pg:
        summarizers[name(method, g)] = partial(train, documents, **g)
    return summarizers

def name(method, instance):
    m = method
    for parameter,value in instance.items():
        m += "-" + parameter + "=" + str(value)
    return m
