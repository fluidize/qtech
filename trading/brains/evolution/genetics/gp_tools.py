import ast
import random
from .param_space import registered_param_specs, ParamSpec

import trading.technical_analysis as ta

def get_indicators(exclude: list[str] = ["hma", "percent_rank", "ichimoku"]):
    """ Get indicator functions from ta module """
    exclude_set = set(exclude)
    indicators = []
    for name in registered_param_specs.keys():
        if name not in exclude_set and hasattr(ta, name):
            indicators.append(getattr(ta, name))
    return indicators

def random_comparison_operator():
    """Get random comparison operator."""
    return random.choice([ast.Gt(), ast.Lt()])

def random_composition_operator():
    """Get random composition operator."""
    return random.choice([ast.BitAnd(), ast.BitOr()])

def paramspecs_to_dict(param_specs: list[ParamSpec]) -> dict:
    return {p.parameter_name : p.search_space for p in param_specs}