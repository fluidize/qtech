import ast
import inspect

from rich.panel import Panel
from rich.console import Console
from rich.syntax import Syntax

import numpy as np
import pandas as pd

import sys
sys.path.append("")
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
    console.print(Panel(Syntax(unparsify(ast_node), "python", theme="native")))

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