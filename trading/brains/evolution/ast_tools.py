import ast

from rich.panel import Panel
from rich.console import Console
from rich.syntax import Syntax

COUNTER = 0  # Used to generate unique parameter names per run

def unique_counter() -> str:
    """Generate a unique integer string."""
    global COUNTER
    name = f"{COUNTER}"
    COUNTER += 1
    return name

def unparsify(ast_node: ast.AST):
    """Convert AST node to Python code string."""
    console = Console()
    ast.fix_missing_locations(ast_node)
    console.print(Panel(Syntax(ast.unparse(ast_node), "python", theme="native")))

def make_compare(left: ast.expr, operator: ast.cmpop, right: ast.expr) -> ast.Compare:
    """Create AST Compare node."""
    return ast.Compare(
        left=left,
        ops=[operator],
        comparators=[right]
    )