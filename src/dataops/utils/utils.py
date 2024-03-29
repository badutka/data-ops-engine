from sklearn.utils import estimator_html_repr
from sklearn import set_config
import seaborn as sns
from functools import wraps
import yaml
from typing import List, AnyStr, Dict, Iterable, Sequence, Mapping
import inspect
import matplotlib.pyplot as plt


def get_lineno():
    return inspect.currentframe().f_back.f_lineno


def read_yaml(filename: AnyStr) -> Dict:
    with open(filename, 'r') as f:
        data_loaded = yaml.safe_load(f)
    return data_loaded


def setup_param_grid(models_params, name):
    model_params = models_params[name]
    param_grid = get_param_grid(model_params)
    param_grid = make_pipeline_grid_names(name, param_grid)
    return param_grid


def get_param_grid(model_params):
    match model_params:
        case {'random_search': d} if d:
            return model_params['random_search']
        case {'grid_search': d} if d:
            return model_params['grid_search']
        case other:
            return {}


def make_pipeline_grid_names(name, param_grid):
    new_param_grid = {}
    for param_name, param_value in param_grid.items():
        new_param_grid[name + "__" + param_name] = param_value
    return new_param_grid


def pipeline_to_html(pipeline, display='diagram', write='a', name="pipeline.html", enc='utf-8'):
    set_config(display=display)
    with open(name, write, encoding=enc) as f:
        f.write(estimator_html_repr(pipeline))


def set_sns_font(font_size):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sns.set(font_scale=font_size)
            rv = func(*args, **kwargs)
            sns.set(font_scale=1)
            return rv

        return wrapper

    return decorator


def set_plot_size(plot_size):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # sns.set(rc={'figure.figsize': plot_size})
            plt.figure(figsize=plot_size)
            rv = func(*args, **kwargs)
            return rv

        return wrapper

    return decorator
