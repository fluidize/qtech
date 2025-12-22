#two indicators two logic form
import ast

from ast_builder import Builder, unparsify
from genes import IndicatorGene, IndicatorToConstant, IndicatorToPrice, IndicatorToIndicator

import sys
sys.path.append("")
import trading.technical_analysis as ta

from rich import print

indicator_genes = [IndicatorGene(function=ta.ema), IndicatorGene(function=ta.rsi)]
logic_genes = [IndicatorToPrice(left_index=0, column_index=3, operator=ast.Gt()), IndicatorToConstant(left_index=1, operator=ast.Gt(), constant=50)]
builder = Builder(indicator_genes=indicator_genes, logic_genes=logic_genes)

base_ast, algorithm_parameter_specs = builder._construct_algorithm_base()
print(unparsify(base_ast))
print(algorithm_parameter_specs)