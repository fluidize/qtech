import pandas as pd
import ast

from rich.panel import Panel
from rich.console import Console
from rich.syntax import Syntax

from genes import IndicatorGene, IndicatorToPrice, IndicatorToConstant, IndicatorToIndicator, LogicGene

class Builder:
    def __init__(self, indicator_genes: list[IndicatorGene], logic_genes: list[LogicGene]):
        self.indicator_genes = indicator_genes
        self.logic_genes = logic_genes
        self.function = None
    
    def _construct_algorithm_base(self):
        algorithm_parameter_specs = [] #all algorithm parameter search spaces to be fed into bayes opt engine

        indicator_ast_list = []
        indicator_variable_names = []
        for gene in self.indicator_genes:
            indicator_ast_list.append(gene.to_ast())
            indicator_variable_names.extend(gene.get_names())
            algorithm_parameter_specs.extend(gene.get_parameter_specs())

        logic_ast_list = []
        logic_variable_names = []
        for gene in self.logic_genes:
            gene.load_indicators(indicator_variable_names)
            logic_ast_list.append(gene.to_ast())
            logic_variable_names.append(gene.get_name())
            algorithm_parameter_specs.extend(gene.get_parameter_specs())

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

        #start off with simple logic assignment
        #signals[logic_variable_names[0]] = 3
        #signals[logic_variable_names[1]] = 2
        buy_conditions_assign = ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id="signals", ctx=ast.Load()), 
                slice=ast.Name(id=logic_variable_names[0], ctx=ast.Load()), 
                ctx=ast.Store()
            )],
            value=ast.Constant(value=3)
        )
        sell_conditions_assign = ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id="signals", ctx=ast.Load()), 
                slice=ast.Name(id=logic_variable_names[1], ctx=ast.Load()), 
                ctx=ast.Store()
            )],
            value=ast.Constant(value=2)
        )

        #return signals
        return_signals = ast.Return(value=ast.Name(id="signals", ctx=ast.Load()))

        body = [
            signals_init,
            *indicator_ast_list,
            *logic_ast_list,
            buy_conditions_assign,
            sell_conditions_assign,
            return_signals
        ]

        args = [ast.arg(arg="data")] + [ast.arg(arg=sp.parameter_name) for sp in algorithm_parameter_specs]
        func_args = ast.arguments(posonlyargs=[], args=args, kwonlyargs=[], kw_defaults=[], defaults=[])
        base_ast = ast.FunctionDef(name="strategy", args=func_args, body=body, decorator_list=[], type_ignores=[]) #default func name is strategy

        return base_ast, algorithm_parameter_specs

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        pass