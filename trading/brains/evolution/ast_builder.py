import pandas as pd
import ast
from rich import print

from genes import IndicatorGene, IndicatorToPrice, IndicatorToConstant, IndicatorToIndicator, LogicGene

import sys
sys.path.append("")
import trading.technical_analysis as ta

def unparsify(ast_node: ast.AST) -> str:
    """Convert AST node to Python code string."""
    ast.fix_missing_locations(ast_node)
    return ast.unparse(ast_node)

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
            indicator_variable_names.append(gene.get_name())
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

        base_ast = ast.Module(body=[signals_init] + indicator_ast_list + logic_ast_list, type_ignores=[])

        return base_ast, algorithm_parameter_specs

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        pass