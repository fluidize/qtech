import ast
from param_space import registered_param_specs
import random

import sys
sys.path.append("")
import trading.technical_analysis as ta

def get_available_indicators():
    """Get all indicator functions from ta module."""
    indicators = []
    for name in registered_param_specs.keys():
        if hasattr(ta, name):
            indicators.append(getattr(ta, name))
    return indicators

def random_operator():
    """Get random comparison operator."""
    return random.choice([ast.Gt(), ast.Lt()])