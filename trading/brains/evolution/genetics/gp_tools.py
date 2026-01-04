import ast
import random

from rich.panel import Panel
from rich.console import Console
from rich.syntax import Syntax

import numpy as np
import pandas as pd

from .param_space import registered_param_specs, ParamSpec

import trading.technical_analysis as ta

COUNTER = 0  # Used to generate unique parameter names per run

def unique_counter() -> str:
    """Generate a unique integer string."""
    global COUNTER
    name = f"{COUNTER}"
    COUNTER += 1
    return name

def unparsify(ast_node: ast.AST):
    """Convert AST node to Python code string."""
    ast.fix_missing_locations(ast_node)
    return ast.unparse(ast_node)

def display_ast(ast_node: ast.AST):
    console = Console()
    syntax = Syntax(
        unparsify(ast_node), 
        "python", 
        theme="monokai",
        word_wrap=True, 
        line_numbers=True,
        indent_guides=True,
        padding=(1, 2)
    )
    console.print(Panel(syntax, border_style="cyan"))

def ast_to_function(ast_node: ast.AST) -> callable:
    """Convert AST node to Python function."""
    module = ast.Module(body=[ast_node], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, "<ast>", "exec")
    namespace = {}
    exec(code, {"np": np, "pd": pd, "ta": ta}, namespace)
    return namespace[ast_node.name]

def make_compare(left: ast.expr, operator: ast.cmpop, right: ast.expr) -> ast.Compare:
    """Create AST Compare node."""
    return ast.Compare(
        left=left,
        ops=[operator],
        comparators=[right]
    )

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