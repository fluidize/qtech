import pandas as pd
import numpy as np
import inspect
import ast

from param_space import registered_param_specs

import sys
sys.path.append("")
import trading.technical_analysis as ta

FUNCTIONAL_ALIAS = "ta"
COUNTER = 0  # Used to generate unique parameter names per run

def unique_counter() -> str:
    """Generate a unique integer string."""
    global COUNTER
    name = f"{COUNTER}"
    COUNTER += 1
    return name

def make_compare(left: ast.expr, operator: ast.cmpop, right: ast.expr) -> ast.Compare:
    """Create AST Compare node."""
    return ast.Compare(
        left=left,
        ops=[operator],
        comparators=[right]
    )

class IndicatorGene:
    """Creates a variable that stores an indicator function."""
    def __init__(self, function: callable):
        self.variable_name = f"{function.__name__}_{unique_counter()}"
        self.function = function
        self.parameter_specs = []
    
    def _get_keywords(self) -> list[ast.expr]:
        """Get the data arguments to be passed to the indicator function."""
        sig = inspect.signature(self.function)
        keywords = []

        data_keywords = {
            "series" : "Close", #by default map all data to the close col
            "high" : "High",
            "low" : "Low",
            "open" : "Open",
            "close" : "Close",
            "volume" : "Volume"
        }
        
        for param_name, param in sig.parameters.items():
            if param_name in data_keywords: #if the parameter should be mapped to a data column
                column = data_keywords[param_name] #column to be accessed from the main df
                keywords.append(
                    ast.keyword(
                        arg=param_name,
                        value=ast.Subscript(
                            value=ast.Name(id="data", ctx=ast.Load()),
                            slice=ast.Constant(value=column),
                            ctx=ast.Load()
                        )
                    )
                )
            else:
                unique_param_name = f"{self.variable_name}_{param_name}"
                keywords.append(
                    ast.keyword(
                        arg=param_name,
                        value=ast.Name(id=unique_param_name, ctx=ast.Load())
                    )
                )
                #add the unique parameter name to the list of parameter names that need to be FILLED OUT on compilation and exec
                self.parameter_specs.append({unique_param_name : registered_param_specs[self.function.__name__][param_name]})
        return keywords
    
    def get_name(self):
        return self.variable_name
    
    def get_parameter_specs(self):
        return self.parameter_specs

    def to_ast(self):
        return ast.Assign(
            targets=[
                ast.Name(
                    id=self.variable_name,
                    ctx=ast.Store()
                )
            ],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=FUNCTIONAL_ALIAS, ctx=ast.Load()), #module alias
                    attr=self.function.__name__,
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[keyword for keyword in self._get_keywords()]
            )
        )

class LogicGene:
    """Creates a variable that stores the result of a logical function."""
    def __init__(self):
        raise NotImplementedError

    def load_indicators(self, indicator_variable_names: list[str]):
        raise NotImplementedError
    
    def get_name(self):
        raise NotImplementedError
    
    def get_parameter_specs(self):
        raise NotImplementedError

    def to_ast(self):
        raise NotImplementedError

class IndicatorToIndicator(LogicGene):
    def __init__(self, left_index: int, right_index: int, operator: ast.cmpop):
        self.left_index = left_index
        self.right_index = right_index
        self.operator = operator
        self.variable_name = None
        self.left_indicator_variable_name = None
        self.right_indicator_variable_name = None
    
    def load_indicators(self, indicator_variable_names: list[str]):
        self.left_indicator_variable_name = indicator_variable_names[self.left_index]
        self.right_indicator_variable_name = indicator_variable_names[self.right_index]
        self.variable_name = f"LOGIC_{self.left_indicator_variable_name}_{self.right_indicator_variable_name}"
    
    def get_name(self):
        return self.variable_name
    
    def get_parameter_specs(self):
        return []
    
    def to_ast(self):
        if self.left_indicator_variable_name is None or self.right_indicator_variable_name is None:
            raise ValueError("Indicators not loaded")
        compare_ast = make_compare(ast.Name(id=self.left_indicator_variable_name, ctx=ast.Load()), self.operator, ast.Name(id=self.right_indicator_variable_name, ctx=ast.Load()))
        return ast.Assign(
            targets=[
                ast.Name(id=self.variable_name, ctx=ast.Store())
            ],
            value=compare_ast
        )

class IndicatorToPrice(LogicGene):
    columns = ["Open", "High", "Low", "Close"]
    def __init__(self, left_index: int, column_index: int, operator: ast.cmpop):
        self.left_index = left_index
        self.column_index = column_index
        self.operator = operator
        self.variable_name = None
        self.left_indicator_variable_name = None
    
    def load_indicators(self, indicator_variable_names: list[str]):
        self.left_indicator_variable_name = indicator_variable_names[self.left_index]
        self.variable_name = f"LOGIC_{self.left_indicator_variable_name}_{self.columns[self.column_index]}"
    
    def get_name(self):
        return self.variable_name
    
    def get_parameter_specs(self):
        return []
    
    def to_ast(self):
        if self.left_indicator_variable_name is None:
            raise ValueError("Indicators not loaded")
        compare_ast = make_compare(ast.Name(id=self.left_indicator_variable_name, ctx=ast.Load()), self.operator, ast.Subscript(
            value=ast.Name(id="data", ctx=ast.Load()),
            slice=ast.Constant(value=self.columns[self.column_index]),
            ctx=ast.Load()
        ))
        return ast.Assign(
            targets=[
                ast.Name(id=self.variable_name, ctx=ast.Store())
            ],
            value=compare_ast
        )

class IndicatorToConstant(LogicGene):
    def __init__(self, left_index: int, operator: ast.cmpop, constant: float):
        self.left_index = left_index
        self.constant = constant
        self.operator = operator
        self.variable_name = None
        self.left_indicator_variable_name = None
        self.parameter_specs = []
    
    def load_indicators(self, indicator_variable_names: list[str]):
        self.left_indicator_variable_name = indicator_variable_names[self.left_index]
        self.variable_name = f"LOGIC_{self.left_indicator_variable_name}_{self.constant}"
        self.parameter_specs = [{f"{self.variable_name}_constant" : (-100, 100)}] #default to this range as constants can be compared to anything
    
    def get_name(self):
        return self.variable_name

    def get_parameter_specs(self):
        return self.parameter_specs

    def to_ast(self):
        if self.left_indicator_variable_name is None:
            raise ValueError("Indicators not loaded")
        compare_ast = make_compare(ast.Name(id=self.left_indicator_variable_name, ctx=ast.Load()), self.operator, ast.Name(id=f"{self.variable_name}_constant", ctx=ast.Load()))
        return ast.Assign(
            targets=[
                ast.Name(id=self.variable_name, ctx=ast.Store())
            ],
            value=compare_ast
        )