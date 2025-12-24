import inspect
import ast

from param_space import registered_param_specs, ParamSpec
from ast_tools import unique_counter, make_compare

FUNCTIONAL_ALIAS = "ta" #module alias for technical analysis functions

class IndicatorGene:
    """Creates a variable that stores an indicator function."""
    def __init__(self, function: callable):
        self.function = function
        self.function_spec = registered_param_specs[self.function.__name__]
        self.parameter_specs = self.function_spec.parameters
        self.unique_parameter_specs = []
        self.variable_names = [f"{self.function.__name__}_{unique_counter()}" for _ in range(self.function_spec.return_count)] #len = return_count
    
    def _get_keywords(self) -> list[ast.expr]:
        """Get the data arguments to be passed to the indicator function."""
        sig = inspect.signature(self.function)
        keywords = []

        data_keywords = {
            "series" : "Close", #by default map all data to the close col
            "high" : "High",
            "low" : "Low",
            "open_price" : "Open", #open is a builtin function, use open_price instead in TA
            "close" : "Close",
            "volume" : "Volume"
        }
        
        for param_name in sig.parameters.keys():
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
                unique_param_name = f"{self.variable_names[0]}_{param_name}"
                keywords.append(
                    ast.keyword(
                        arg=param_name,
                        value=ast.Name(id=unique_param_name, ctx=ast.Load())
                    )
                )
                current_param_spec = next((p for p in self.parameter_specs if p.parameter_name == param_name), None)
                self.unique_parameter_specs.append(ParamSpec(parameter_name=unique_param_name, search_space=current_param_spec.search_space))
        return keywords
    
    def get_names(self):
        return self.variable_names
    
    def get_parameter_specs(self):
        return self.unique_parameter_specs

    def to_ast(self):
        if self.function_spec.return_count > 1:
            targets = ast.Tuple(elts=[ast.Name(id=variable_name, ctx=ast.Store()) for variable_name in self.variable_names], ctx=ast.Store())
        else:
            targets = ast.Name(id=self.variable_names[0], ctx=ast.Store())
        return ast.Assign(
            targets=[targets],
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
        self.parameter_specs = None
    
    def load_indicators(self, indicator_variable_names: list[str]):
        self.left_indicator_variable_name = indicator_variable_names[self.left_index]
        self.variable_name = f"LOGIC_{self.left_indicator_variable_name}_const_{unique_counter()}"
        self.parameter_specs = [ParamSpec(parameter_name=f"{self.variable_name}_constant", search_space=(-100, 100))]
    
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