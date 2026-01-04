import pandas as pd
import ast
import inspect
import copy

from .param_space import registered_param_specs, ParamSpec
from .gp_tools import unique_counter, make_compare, ast_to_function, paramspecs_to_dict

FUNCTIONAL_ALIAS = "ta" #module alias for technical analysis functions

class IndicatorGene:
    def __init__(self, function: callable):
        self.function = function
        self.function_spec = registered_param_specs[self.function.__name__]
        self.parameter_specs = self.function_spec.parameters
        self.unique_parameter_specs = []
        self.variable_names = [f"{self.function.__name__}{unique_counter()}" for _ in range(self.function_spec.return_count)] #len = return_count
    
    def _get_keywords(self) -> list[ast.expr]:
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
                if unique_param_name not in {sp.parameter_name for sp in self.unique_parameter_specs}:
                    current_param_spec = next((p for p in self.parameter_specs if p.parameter_name == param_name), None)
                    if current_param_spec:
                        self.unique_parameter_specs.append(ParamSpec(parameter_name=unique_param_name, search_space=current_param_spec.search_space))
        return keywords
    
    def get_names(self):
        return self.variable_names
    
    def get_names_flattened(self):
        for name in self.get_names():
            yield from name
    
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
    
    def __eq__(self, other):
        if not isinstance(other, IndicatorGene):
            return False
        return self.function.__name__ == other.function.__name__
    
    def __hash__(self):
        return hash(self.function.__name__)

class LogicGene:
    def __init__(self):
        raise NotImplementedError
    
    def get_name(self):
        raise NotImplementedError
    
    def get_parameter_specs(self):
        raise NotImplementedError

    def to_ast(self):
        raise NotImplementedError

#i2i
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
        self.variable_name = f"LOGIC_{self.left_indicator_variable_name}_{type(self.operator).__name__}_{self.right_indicator_variable_name}" if self.variable_name is None else self.variable_name
    
    def get_name(self):
        return self.variable_name
    
    def get_parameter_specs(self):
        return []

    def get_referenced_indicator_names(self):
        if self.left_indicator_variable_name is None or self.right_indicator_variable_name is None:
            return []
        return [self.left_indicator_variable_name, self.right_indicator_variable_name]
    
    def get_referenced_indicators(self, sequence_dict: dict):
        return [sequence_dict[name] for name in self.get_referenced_indicator_names() if name in sequence_dict]

    def set_index(self, left_index: int, right_index: int):
        self.left_index = left_index
        self.right_index = right_index
        self.left_indicator_variable_name = None
        self.right_indicator_variable_name = None

    def to_ast(self):
        if self.left_indicator_variable_name is None or self.right_indicator_variable_name is None:
            raise ValueError("Indicators not loaded - call load_indicators() first")
        compare_ast = make_compare(ast.Name(id=self.left_indicator_variable_name, ctx=ast.Load()), self.operator, ast.Name(id=self.right_indicator_variable_name, ctx=ast.Load()))
        return ast.Assign(
            targets=[
                ast.Name(id=self.variable_name, ctx=ast.Store())
            ],
            value=compare_ast
        )
    
    def __eq__(self, other):
        if not isinstance(other, IndicatorToIndicator):
            return False
        return (self.left_index == other.left_index and 
                self.right_index == other.right_index and
                type(self.operator) == type(other.operator))
    
    def __hash__(self):
        return hash((self.left_index, self.right_index, type(self.operator)))

#i2p
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
        self.variable_name = f"LOGIC_{self.left_indicator_variable_name}_{type(self.operator).__name__}_{self.columns[self.column_index]}" if self.variable_name is None else self.variable_name
    
    def get_name(self):
        return self.variable_name
    
    def get_parameter_specs(self):
        return []

    def get_referenced_indicator_name(self):
        if self.left_indicator_variable_name is None:
            return None
        return self.left_indicator_variable_name
    
    def get_referenced_indicator(self, sequence_dict: dict):
        return sequence_dict[self.get_referenced_indicator_name()]

    def set_index(self, left_index: int):
        self.left_index = left_index
        self.left_indicator_variable_name = None

    def to_ast(self):
        if self.left_indicator_variable_name is None:
            raise ValueError("Indicators not loaded - call load_indicators() first")
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
    
    def __eq__(self, other):
        if not isinstance(other, IndicatorToPrice):
            return False
        return (self.left_index == other.left_index and
                self.column_index == other.column_index and
                type(self.operator) == type(other.operator))
    
    def __hash__(self):
        return hash((self.left_index, self.column_index, type(self.operator)))

#i2c
class IndicatorToConstant(LogicGene):
    def __init__(self, left_index: int, operator: ast.cmpop):
        self.left_index = left_index
        self.operator = operator
        self.variable_name = None
        self.left_indicator_variable_name = None
        self.parameter_specs = None
    
    def load_indicators(self, indicator_variable_names: list[str]):
        self.left_indicator_variable_name = indicator_variable_names[self.left_index]
        self.variable_name = f"LOGIC_{self.left_indicator_variable_name}_{type(self.operator).__name__}_const_{unique_counter()}" if self.variable_name is None else self.variable_name
        self.parameter_specs = [ParamSpec(parameter_name=f"{self.variable_name}_constant", search_space=(-100, 100))]
    
    def get_name(self):
        return self.variable_name

    def get_parameter_specs(self):
        return self.parameter_specs

    def get_referenced_indicator_name(self):
        if self.left_indicator_variable_name is None:
            return None
        return self.left_indicator_variable_name
    
    def get_referenced_indicator(self, sequence_dict: dict):
        return sequence_dict[self.get_referenced_indicator_name()]

    def set_index(self, left_index: int, right_index: int = None):
        self.left_index = left_index
        self.left_indicator_variable_name = None
        # Note: parameter_specs will be regenerated in load_indicators() if needed

    def to_ast(self):
        if self.left_indicator_variable_name is None:
            raise ValueError("Indicators not loaded - call load_indicators() first")
        compare_ast = make_compare(ast.Name(id=self.left_indicator_variable_name, ctx=ast.Load()), self.operator, ast.Name(id=f"{self.variable_name}_constant", ctx=ast.Load()))
        return ast.Assign(
            targets=[
                ast.Name(id=self.variable_name, ctx=ast.Store())
            ],
            value=compare_ast
        )
    
    def __eq__(self, other):
        if not isinstance(other, IndicatorToConstant):
            return False
        return (self.left_index == other.left_index and
                type(self.operator) == type(other.operator))
    
    def __hash__(self):
        return hash((self.left_index, type(self.operator)))

#l2l
class LogicToLogic(LogicGene):
    def __init__(self, left_logic_index: int, right_logic_index: int, operator: ast.cmpop):
        self.left_logic_index = left_logic_index
        self.right_logic_index = right_logic_index
        self.operator = operator
        self.variable_name = None
        self.left_logic_variable_name = None
        self.right_logic_variable_name = None
        self.logic_variable_names = None
    
    def load_logic(self, logic_variable_names: list[str]):
        self.logic_variable_names = logic_variable_names
        self.left_logic_variable_name = self.logic_variable_names[self.left_logic_index]
        self.right_logic_variable_name = self.logic_variable_names[self.right_logic_index]
        self.variable_name = f"LOGIC_COMPOSITE_{unique_counter()}" if self.variable_name is None else self.variable_name
    
    def get_name(self):
        return self.variable_name
    
    def get_parameter_specs(self):
        return []

    def get_referenced_logic_names(self):
        if self.left_logic_variable_name is None or self.right_logic_variable_name is None:
            return []
        return [self.left_logic_variable_name, self.right_logic_variable_name]
    
    def get_referenced_logic(self, sequence_dict: dict):
        return [sequence_dict[name] for name in self.get_referenced_logic_names() if name in sequence_dict]

    def set_index(self, left_logic_index: int, right_logic_index: int):
        self.left_logic_index = left_logic_index
        self.right_logic_index = right_logic_index

    def to_ast(self):
        if self.left_logic_variable_name is None or self.right_logic_variable_name is None:
            raise ValueError("Logic genes not loaded - call load_logic() first")
        
        left_expr = ast.Name(id=self.left_logic_variable_name, ctx=ast.Load())
        right_expr = ast.Name(id=self.right_logic_variable_name, ctx=ast.Load())
        return ast.Assign(
            targets=[ast.Name(id=self.variable_name, ctx=ast.Store())],
            value=ast.BinOp(left=left_expr, op=self.operator, right=right_expr)
        )
    
    def __eq__(self, other):
        if not isinstance(other, LogicToLogic):
            return False
        return (self.left_logic_index == other.left_logic_index and
                self.right_logic_index == other.right_logic_index and
                type(self.operator) == type(other.operator))
    
    def __hash__(self):
        return hash((self.left_logic_index, self.right_logic_index, type(self.operator)))

class SignalGene():
    def __init__(self, long_logic_index: int, short_logic_index: int):
        self.long_logic_index = long_logic_index
        self.short_logic_index = short_logic_index
        self.variable_name = None
        self.long_logic_variable_name = None
        self.short_logic_variable_name = None
    
    def load_logic(self, logic_variable_names: list[str]):
        self.logic_variable_names = logic_variable_names
        self.long_logic_variable_name = logic_variable_names[self.long_logic_index]
        self.short_logic_variable_name = logic_variable_names[self.short_logic_index]
        self.variable_name = f"SIGNAL_{self.long_logic_variable_name}_{self.short_logic_variable_name}" if self.variable_name is None else self.variable_name
    
    def get_name(self):
        return self.variable_name

    def get_referenced_logic_names(self):
        if self.long_logic_variable_name is None or self.short_logic_variable_name is None:
            return []
        return [self.long_logic_variable_name, self.short_logic_variable_name]
    
    def get_referenced_logic(self, sequence_dict: dict):
        return [sequence_dict[name] for name in self.get_referenced_logic_names() if name in sequence_dict]

    def set_index(self, long_logic_index: int, short_logic_index: int):
        self.long_logic_index = long_logic_index
        self.short_logic_index = short_logic_index
        self.long_logic_variable_name = None
        self.short_logic_variable_name = None
    
    def to_ast(self):
        if self.long_logic_variable_name is None or self.short_logic_variable_name is None:
            raise ValueError("Logic genes not loaded - call load_logic() first")
        buy_conditions_assign = ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id="signals", ctx=ast.Load()), 
                slice=ast.Name(id=self.long_logic_variable_name, ctx=ast.Load()), 
                ctx=ast.Store()
            )],
            value=ast.Constant(value=1)
        )
        sell_conditions_assign = ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id="signals", ctx=ast.Load()), 
                slice=ast.Name(id=self.short_logic_variable_name, ctx=ast.Load()), 
                ctx=ast.Store()
            )],
            value=ast.Constant(value=0)
        )
        return [buy_conditions_assign, sell_conditions_assign]
    
    def __eq__(self, other):
        if not isinstance(other, SignalGene):
            return False
        return (self.long_logic_index == other.long_logic_index and
                self.short_logic_index == other.short_logic_index)
    
    def __hash__(self):
        return hash((self.long_logic_index, self.short_logic_index))

class Genome:
    def __init__(self, indicator_genes: list[IndicatorGene], logic_genes: list[LogicGene], signal_genes: list[SignalGene]):
        self.indicator_genes = indicator_genes
        self.logic_genes = logic_genes
        self.signal_genes = signal_genes

        self.build_genome()

    def build_genome(self):
        self.function_ast, self.param_space = self.construct_algorithm()
        self.param_space = paramspecs_to_dict(self.param_space)
        self.compiled_function = ast_to_function(self.function_ast)
        self.compiled_function.param_space = self.param_space

    def construct_algorithm(self):
        algorithm_parameter_specs = [] #all algorithm parameter search spaces to be fed into bayes opt engine

        indicator_ast_list = []
        indicator_variable_names = []
        for gene in self.indicator_genes:
            indicator_ast_list.append(gene.to_ast())
            indicator_variable_names.extend(gene.get_names())
            algorithm_parameter_specs.extend(gene.get_parameter_specs())

        logic_ast_list = []
        logic_variable_names = []
        
        simple_logic_genes = []
        composite_logic_genes = []
        for gene in self.logic_genes:
            if isinstance(gene, LogicToLogic):
                composite_logic_genes.append(gene)
            else:
                simple_logic_genes.append(gene)
        
        for simple_logic_gene in simple_logic_genes:
            simple_logic_gene.load_indicators(indicator_variable_names)
            logic_ast_list.append(simple_logic_gene.to_ast())
            logic_variable_names.append(simple_logic_gene.get_name())
            algorithm_parameter_specs.extend(simple_logic_gene.get_parameter_specs())

        for composite_logic_gene in composite_logic_genes:
            composite_logic_gene.load_logic(logic_variable_names)
            logic_ast_list.append(composite_logic_gene.to_ast())
            logic_variable_names.append(composite_logic_gene.get_name())
            algorithm_parameter_specs.extend(composite_logic_gene.get_parameter_specs())

        for signal_gene in self.signal_genes:
            signal_gene.load_logic(logic_variable_names)

        #signals = pd.Series(0, index=data.index)
        signals_init = ast.Assign(
            targets=[ast.Name(id="signals", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="pd", ctx=ast.Load()),
                    attr="Series",
                    ctx=ast.Load()
                ),
                args=[ast.Constant(value=0)],
                keywords=[
                    ast.keyword(arg="index", value=ast.Attribute(value=ast.Name(id="data", ctx=ast.Load()),attr="index",ctx=ast.Load()))
                ]
            )
        )

        buy_conditions_assign, sell_conditions_assign = zip(*[signal_gene.to_ast() for signal_gene in self.signal_genes])

        #return signals
        return_signals = ast.Return(value=ast.Name(id="signals", ctx=ast.Load()))

        body = [
            signals_init,
            *indicator_ast_list,
            *logic_ast_list,
            *buy_conditions_assign,
            *sell_conditions_assign,
            return_signals
        ]

        args = [ast.arg(arg="data")] + [ast.arg(arg=sp.parameter_name) for sp in algorithm_parameter_specs]
        func_args = ast.arguments(posonlyargs=[], args=args, kwonlyargs=[], kw_defaults=[], defaults=[])
        func_ast = ast.FunctionDef(name="strategy", args=func_args, body=body, decorator_list=[], type_ignores=[]) #default func name is strategy

        return func_ast, algorithm_parameter_specs

    def get_function_ast(self) -> ast.AST:
        return self.function_ast
    
    def get_compiled_function(self) -> callable:
        return self.compiled_function

    def get_param_space(self) -> dict:
        return self.param_space

    def __call__(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return self.compiled_function(data, **kwargs)