import pandas as pd
import ast
import inspect
import random
import numpy as np
from collections import Counter

from .param_space import registered_param_specs, ParamSpec
from .tools import *

FUNCTIONAL_ALIAS = "ta"  # module alias for technical analysis functions
EXCLUDED_INDICATORS = ()
_INDICATORS_CACHE = None


class IndicatorGene:
    def __init__(self, function: callable):
        self.function = function
        self.function_spec = registered_param_specs[self.function.__name__]
        self.parameter_specs = self.function_spec.parameters
        self.unique_parameter_specs = []
        self.variable_name = f"{self.function.__name__}{unique_counter()}"

    def _get_keywords(self) -> list[ast.expr]:
        sig = inspect.signature(self.function)
        keywords = []

        data_keywords = {  # keywords are inside original TA function params
            "series": "Close",  # by default map all data to the close col
            "high": "High",
            "low": "Low",
            "open_price": "Open",  # open is a builtin function, use open_price instead in TA
            "close": "Close",
            "volume": "Volume",
        }

        for param_name in sig.parameters.keys():
            if (
                param_name in data_keywords
            ):  # if the parameter should be mapped to a data column
                column = data_keywords[
                    param_name
                ]  # column to be accessed from the main df
                keywords.append(
                    ast.keyword(
                        arg=param_name,
                        value=ast.Subscript(
                            value=ast.Name(id="data", ctx=ast.Load()),
                            slice=ast.Constant(value=column),
                            ctx=ast.Load(),
                        ),
                    )
                )
            else:
                unique_param_name = f"{self.variable_name}_{param_name}"
                keywords.append(
                    ast.keyword(
                        arg=param_name,
                        value=ast.Name(id=unique_param_name, ctx=ast.Load()),
                    )
                )
                if unique_param_name not in {
                    sp.parameter_name for sp in self.unique_parameter_specs
                }:
                    current_param_spec = next(
                        (
                            p
                            for p in self.parameter_specs
                            if p.parameter_name == param_name
                        ),
                        None,
                    )
                    if current_param_spec:
                        self.unique_parameter_specs.append(
                            ParamSpec(
                                parameter_name=unique_param_name,
                                search_space=current_param_spec.search_space,
                            )
                        )
        return keywords

    def get_name(self):
        return self.variable_name

    def get_parameter_specs(self):
        return self.unique_parameter_specs

    def to_ast(self):
        return ast.Assign(
            targets=[ast.Name(id=self.variable_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=FUNCTIONAL_ALIAS, ctx=ast.Load()),  # module alias
                    attr=self.function.__name__,
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[keyword for keyword in self._get_keywords()],
            ),
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


# i2i
class IndicatorToIndicator(LogicGene):
    def __init__(
        self,
        left_indicator: IndicatorGene,
        right_indicator: IndicatorGene,
        operator: ast.cmpop,
    ):
        self.left_indicator = left_indicator
        self.right_indicator = right_indicator
        self.operator = operator
        self.variable_name = None
        self.left_indicator_variable_name = None
        self.right_indicator_variable_name = None

    def load_indicators(self, indicator_variable_names: list[str]):
        self.left_indicator_variable_name = self.left_indicator.get_name()
        self.right_indicator_variable_name = self.right_indicator.get_name()
        self.variable_name = (
            f"LOGIC_{self.left_indicator_variable_name}_{type(self.operator).__name__}_{self.right_indicator_variable_name}"
            if self.variable_name is None
            else self.variable_name
        )

    def get_name(self):
        return self.variable_name

    def get_parameter_specs(self):
        return []

    def get_referenced_indicator_names(self):
        if (
            self.left_indicator_variable_name is None
            or self.right_indicator_variable_name is None
        ):
            return []
        return [self.left_indicator_variable_name, self.right_indicator_variable_name]

    def get_referenced_indicators(self, sequence_dict: dict):
        return [self.left_indicator, self.right_indicator]

    def to_ast(self):
        if (
            self.left_indicator_variable_name is None
            or self.right_indicator_variable_name is None
        ):
            raise ValueError("Indicators not loaded - call load_indicators() first")
        compare_ast = make_compare(
            ast.Name(id=self.left_indicator_variable_name, ctx=ast.Load()),
            self.operator,
            ast.Name(id=self.right_indicator_variable_name, ctx=ast.Load()),
        )
        return ast.Assign(
            targets=[ast.Name(id=self.variable_name, ctx=ast.Store())],
            value=compare_ast,
        )

    def __eq__(self, other):
        if not isinstance(other, IndicatorToIndicator):
            return False
        return (
            self.left_indicator == other.left_indicator
            and self.right_indicator == other.right_indicator
            and type(self.operator) == type(other.operator)
        )

    def __hash__(self):
        return hash((self.left_indicator, self.right_indicator, type(self.operator)))


# i2p
class IndicatorToPrice(LogicGene):
    columns = ["Open", "High", "Low", "Close"]

    def __init__(
        self, left_indicator: IndicatorGene, column_index: int, operator: ast.cmpop
    ):
        self.left_indicator = left_indicator
        self.column_index = column_index
        self.operator = operator
        self.variable_name = None
        self.left_indicator_variable_name = None

    def load_indicators(self, indicator_variable_names: list[str]):
        self.left_indicator_variable_name = self.left_indicator.get_name()
        self.variable_name = (
            f"LOGIC_{self.left_indicator_variable_name}_{type(self.operator).__name__}_{self.columns[self.column_index]}"
            if self.variable_name is None
            else self.variable_name
        )

    def get_name(self):
        return self.variable_name

    def get_parameter_specs(self):
        return []

    def get_referenced_indicator_name(self):
        if self.left_indicator_variable_name is None:
            return None
        return self.left_indicator_variable_name

    def get_referenced_indicator(self, sequence_dict: dict):
        return self.left_indicator

    def to_ast(self):
        if self.left_indicator_variable_name is None:
            raise ValueError("Indicators not loaded - call load_indicators() first")
        compare_ast = make_compare(
            ast.Name(id=self.left_indicator_variable_name, ctx=ast.Load()),
            self.operator,
            ast.Subscript(
                value=ast.Name(id="data", ctx=ast.Load()),
                slice=ast.Constant(value=self.columns[self.column_index]),
                ctx=ast.Load(),
            ),
        )
        return ast.Assign(
            targets=[ast.Name(id=self.variable_name, ctx=ast.Store())],
            value=compare_ast,
        )

    def __eq__(self, other):
        if not isinstance(other, IndicatorToPrice):
            return False
        return (
            self.left_indicator == other.left_indicator
            and self.column_index == other.column_index
            and type(self.operator) == type(other.operator)
        )

    def __hash__(self):
        return hash((self.left_indicator, self.column_index, type(self.operator)))


# i2c
class IndicatorToConstant(LogicGene):
    def __init__(self, left_indicator: IndicatorGene, operator: ast.cmpop):
        self.left_indicator = left_indicator
        self.operator = operator
        self.variable_name = None
        self.left_indicator_variable_name = None
        self.parameter_specs = None

    def load_indicators(self, indicator_variable_names: list[str]):
        self.left_indicator_variable_name = self.left_indicator.get_name()
        self.variable_name = (
            f"LOGIC_{self.left_indicator_variable_name}_{type(self.operator).__name__}_const_{unique_counter()}"
            if self.variable_name is None
            else self.variable_name
        )
        self.parameter_specs = [
            ParamSpec(
                parameter_name=f"{self.variable_name}_constant",
                search_space=(-100, 100),
            )
        ]

    def get_name(self):
        return self.variable_name

    def get_parameter_specs(self):
        return self.parameter_specs

    def get_referenced_indicator_name(self):
        if self.left_indicator_variable_name is None:
            return None
        return self.left_indicator_variable_name

    def get_referenced_indicator(self, sequence_dict: dict):
        return self.left_indicator

    def to_ast(self):
        if self.left_indicator_variable_name is None:
            raise ValueError("Indicators not loaded - call load_indicators() first")
        compare_ast = make_compare(
            ast.Name(id=self.left_indicator_variable_name, ctx=ast.Load()),
            self.operator,
            ast.Name(id=f"{self.variable_name}_constant", ctx=ast.Load()),
        )
        return ast.Assign(
            targets=[ast.Name(id=self.variable_name, ctx=ast.Store())],
            value=compare_ast,
        )

    def __eq__(self, other):
        if not isinstance(other, IndicatorToConstant):
            return False
        return self.left_indicator == other.left_indicator and type(
            self.operator
        ) == type(other.operator)

    def __hash__(self):
        return hash((self.left_indicator, type(self.operator)))


# l2l
class LogicToLogic(LogicGene):
    def __init__(
        self, left_logic: LogicGene, right_logic: LogicGene, operator: ast.cmpop
    ):
        self.left_logic = left_logic
        self.right_logic = right_logic
        self.operator = operator
        self.variable_name = None
        self.left_logic_variable_name = None
        self.right_logic_variable_name = None

    def load_logic(self, logic_variable_names: list[str]):
        self.left_logic_variable_name = self.left_logic.get_name()
        self.right_logic_variable_name = self.right_logic.get_name()

        self.variable_name = (
            f"LOGIC_COMPOSITE_{unique_counter()}"
            if self.variable_name is None
            else self.variable_name
        )

    def get_name(self):
        return self.variable_name

    def get_parameter_specs(self):
        return []

    def get_referenced_logic_names(self):
        if (
            self.left_logic_variable_name is None
            or self.right_logic_variable_name is None
        ):
            return []
        return [self.left_logic_variable_name, self.right_logic_variable_name]

    def get_referenced_logic(self, sequence_dict: dict):
        return [self.left_logic, self.right_logic]

    def to_ast(self):
        if (
            self.left_logic_variable_name is None
            or self.right_logic_variable_name is None
        ):
            raise ValueError("Logic genes not loaded - call load_logic() first")

        left_expr = ast.Name(id=self.left_logic_variable_name, ctx=ast.Load())
        right_expr = ast.Name(id=self.right_logic_variable_name, ctx=ast.Load())
        return ast.Assign(
            targets=[ast.Name(id=self.variable_name, ctx=ast.Store())],
            value=ast.BinOp(left=left_expr, op=self.operator, right=right_expr),
        )

    def __eq__(self, other):
        if not isinstance(other, LogicToLogic):
            return False
        return (
            self.left_logic == other.left_logic
            and self.right_logic == other.right_logic
            and type(self.operator) == type(other.operator)
        )

    def __hash__(self):
        return hash((self.left_logic, self.right_logic, type(self.operator)))


class SignalGene:
    def __init__(self, long_logic: LogicGene, short_logic: LogicGene):
        self.long_logic = long_logic
        self.short_logic = short_logic
        self.variable_name = None
        self.long_logic_variable_name = None
        self.short_logic_variable_name = None

    def load_logic(self, logic_variable_names: list[str]):
        self.long_logic_variable_name = self.long_logic.get_name()
        self.short_logic_variable_name = self.short_logic.get_name()
        self.variable_name = (
            f"SIGNAL_{self.long_logic_variable_name}_{self.short_logic_variable_name}"
            if self.variable_name is None
            else self.variable_name
        )

    def get_name(self):
        return self.variable_name

    def get_referenced_logic_names(self):
        if (
            self.long_logic_variable_name is None
            or self.short_logic_variable_name is None
        ):
            return []
        return [self.long_logic_variable_name, self.short_logic_variable_name]

    def get_referenced_logic(self, sequence_dict: dict):
        return [self.long_logic, self.short_logic]

    def to_ast(self):
        if (
            self.long_logic_variable_name is None
            or self.short_logic_variable_name is None
        ):
            raise ValueError("Logic genes not loaded - call load_logic() first")
        buy_conditions_assign = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id="signals", ctx=ast.Load()),
                    slice=ast.Name(id=self.long_logic_variable_name, ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=1),
        )
        sell_conditions_assign = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id="signals", ctx=ast.Load()),
                    slice=ast.Name(id=self.short_logic_variable_name, ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=0),
        )
        return [buy_conditions_assign, sell_conditions_assign]

    def __eq__(self, other):
        if not isinstance(other, SignalGene):
            return False
        return (
            self.long_logic == other.long_logic
            and self.short_logic == other.short_logic
        )

    def __hash__(self):
        return hash((self.long_logic, self.short_logic))


def _get_indicators_cached():
    global _INDICATORS_CACHE
    if _INDICATORS_CACHE is None:
        _INDICATORS_CACHE = get_indicators(exclude=EXCLUDED_INDICATORS)
    return _INDICATORS_CACHE


def generate_indicator_gene() -> IndicatorGene:
    return IndicatorGene(function=random.choice(_get_indicators_cached()))


def generate_logic_gene_sequence(
    num_logic: int,
    indicator_genes: list[IndicatorGene],
    allow_logic_composition: bool,
    logic_composition_prob: float,
    existing_logic_genes: list[LogicGene] = None,
) -> list[LogicGene]:
    logic_genes = []
    for _ in range(num_logic):
        if (
            allow_logic_composition
            and existing_logic_genes is not None
            and len(existing_logic_genes) >= 2
            and random.random() < logic_composition_prob
        ):
            left_logic, right_logic = random.sample(existing_logic_genes, 2)
            logic_genes.append(
                LogicToLogic(
                    left_logic=left_logic,
                    right_logic=right_logic,
                    operator=random_composition_operator(),
                )
            )
        else:
            logic_type = random.choice(
                [IndicatorToPrice, IndicatorToConstant, IndicatorToIndicator]
            )

            if logic_type == IndicatorToPrice:
                logic_genes.append(
                    IndicatorToPrice(
                        left_indicator=random.choice(indicator_genes),
                        column_index=random.randint(0, 3),
                        operator=random_comparison_operator(),
                    )
                )
            elif logic_type == IndicatorToConstant:
                logic_genes.append(
                    IndicatorToConstant(
                        left_indicator=random.choice(indicator_genes),
                        operator=random_comparison_operator(),
                    )
                )
            else:
                left_indicator = random.choice(indicator_genes)
                right_indicator = random.choice(indicator_genes)
                while right_indicator == left_indicator and len(indicator_genes) > 1:
                    right_indicator = random.choice(indicator_genes)
                logic_genes.append(
                    IndicatorToIndicator(
                        left_indicator=left_indicator,
                        right_indicator=right_indicator,
                        operator=random_comparison_operator(),
                    )
                )
    return logic_genes


def generate_signal_gene(logic_genes: list[LogicGene]) -> SignalGene:
    long_logic, short_logic = random.sample(logic_genes, 2)
    return SignalGene(long_logic=long_logic, short_logic=short_logic)


def generate_genome(
    num_indicators=None,
    num_logic=None,
    min_indicators=2,
    max_indicators=6,
    min_logic=2,
    max_logic=6,
    allow_logic_composition=True,
    logic_composition_prob=0.5,
):
    if num_indicators is None:
        num_indicators = random.randint(min_indicators, max_indicators)
    if num_logic is None:
        num_logic = random.randint(min_logic, max_logic)

    indicator_genes = [generate_indicator_gene() for _ in range(num_indicators)]

    logic_genes = generate_logic_gene_sequence(
        num_logic,
        indicator_genes,
        allow_logic_composition,
        logic_composition_prob,
    )
    signal_genes = [generate_signal_gene(logic_genes)]
    return Genome(
        indicator_genes=indicator_genes,
        logic_genes=logic_genes,
        signal_genes=signal_genes,
    )


def generate_population(
    size=100,
    num_indicators=None,
    num_logic=None,
    min_indicators=2,
    max_indicators=6,
    min_logic=2,
    max_logic=6,
    allow_logic_composition=True,
    logic_composition_prob=0.5,
):
    population = []
    for _ in range(size):
        genome = generate_genome(
            num_indicators=num_indicators,
            num_logic=num_logic,
            min_indicators=min_indicators,
            max_indicators=max_indicators,
            min_logic=min_logic,
            max_logic=max_logic,
            allow_logic_composition=allow_logic_composition,
            logic_composition_prob=logic_composition_prob,
        )
        population.append(genome)
    return population


def separate_genes(
    genes_list: list,
) -> tuple[list[IndicatorGene], list[LogicGene], list[SignalGene]]:
    indicator_genes = []
    logic_genes = []
    signal_genes = []

    for gene in genes_list:
        if isinstance(gene, IndicatorGene):
            indicator_genes.append(gene)
        elif isinstance(gene, LogicGene):
            logic_genes.append(gene)
        elif isinstance(gene, SignalGene):
            signal_genes.append(gene)

    return indicator_genes, logic_genes, signal_genes


class Genome:
    def __init__(
        self,
        indicator_genes: list[IndicatorGene],
        logic_genes: list[LogicGene],
        signal_genes: list[SignalGene],
    ):
        self.indicator_genes = indicator_genes
        self.logic_genes = logic_genes
        self.signal_genes = signal_genes

        indicator_ast_list, logic_ast_list, algorithm_parameter_specs = (
            self._prepare_genes()
        )

        self.function_ast, self.param_space = self._construct_algorithm(
            indicator_ast_list=indicator_ast_list,
            logic_ast_list=logic_ast_list,
            algorithm_parameter_specs=algorithm_parameter_specs,
        )
        self.param_space = paramspecs_to_dict(self.param_space)
        self.compiled_function = ast_to_function(self.function_ast)
        self.compiled_function.param_space = self.param_space

        self.sequence_dict = {}
        for gene in self.indicator_genes + self.logic_genes + self.signal_genes:
            if isinstance(gene, IndicatorGene):
                self.sequence_dict[gene.get_name()] = gene  # Now single string
            else:
                self.sequence_dict[gene.get_name()] = gene

        self._best_params = None
        self._best_metric = None

    def _prepare_genes(self):
        reset_counter()
        algorithm_parameter_specs = (
            []
        )  # all algorithm parameter search spaces to be fed into bayes opt engine

        indicator_ast_list = []
        indicator_variable_names = []
        for gene in self.indicator_genes:
            indicator_ast_list.append(gene.to_ast())
            indicator_variable_names.append(
                gene.get_name()
            )  # Now single string, not list
            algorithm_parameter_specs.extend(gene.get_parameter_specs())

        logic_ast_list = []
        logic_variable_names = []

        for logic_gene in self.logic_genes:
            if isinstance(logic_gene, LogicToLogic):
                logic_gene.load_logic(logic_variable_names)
            else:
                logic_gene.load_indicators(indicator_variable_names)
            logic_variable_names.append(logic_gene.get_name())
            logic_ast_list.append(logic_gene.to_ast())
            algorithm_parameter_specs.extend(logic_gene.get_parameter_specs())

        for signal_gene in self.signal_genes:
            signal_gene.load_logic(logic_variable_names)

        return indicator_ast_list, logic_ast_list, algorithm_parameter_specs

    def _construct_algorithm(
        self, indicator_ast_list, logic_ast_list, algorithm_parameter_specs
    ):
        # signals = pd.Series(0, index=data.index)
        signals_init = ast.Assign(
            targets=[ast.Name(id="signals", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="pd", ctx=ast.Load()),
                    attr="Series",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=0)],
                keywords=[
                    ast.keyword(
                        arg="index",
                        value=ast.Attribute(
                            value=ast.Name(id="data", ctx=ast.Load()),
                            attr="index",
                            ctx=ast.Load(),
                        ),
                    )
                ],
            ),
        )

        buy_conditions_assign, sell_conditions_assign = zip(
            *[signal_gene.to_ast() for signal_gene in self.signal_genes]
        )

        # return signals
        return_signals = ast.Return(value=ast.Name(id="signals", ctx=ast.Load()))

        body = [
            signals_init,
            *indicator_ast_list,
            *logic_ast_list,
            *buy_conditions_assign,
            *sell_conditions_assign,
            return_signals,
        ]

        args = [ast.arg(arg="data")] + [
            ast.arg(arg=sp.parameter_name) for sp in algorithm_parameter_specs
        ]
        func_args = ast.arguments(
            posonlyargs=[], args=args, kwonlyargs=[], kw_defaults=[], defaults=[]
        )
        func_ast = ast.FunctionDef(
            name="strategy",
            args=func_args,
            body=body,
            decorator_list=[],
            type_ignores=[],
        )  # default func name is strategy

        return func_ast, algorithm_parameter_specs

    def remove_unused_genes(self):
        used_logic_genes = []
        for gene in self.signal_genes:
            referenced = gene.get_referenced_logic(self.sequence_dict)
            for ref_gene in referenced:
                if ref_gene not in used_logic_genes:
                    used_logic_genes.append(ref_gene)

        i = 0
        while i < len(used_logic_genes):
            gene = used_logic_genes[i]
            if isinstance(gene, LogicToLogic):
                referenced = gene.get_referenced_logic(self.sequence_dict)
                for ref_gene in referenced:
                    if ref_gene not in used_logic_genes:
                        used_logic_genes.append(ref_gene)
            i += 1

        used_indicator_genes = []
        for gene in used_logic_genes:
            if isinstance(gene, IndicatorToIndicator):
                referenced = gene.get_referenced_indicators(self.sequence_dict)
                for ref_gene in referenced:
                    if ref_gene not in used_indicator_genes:
                        used_indicator_genes.append(ref_gene)
            elif isinstance(gene, IndicatorToPrice) or isinstance(
                gene, IndicatorToConstant
            ):
                ref_gene = gene.get_referenced_indicator(self.sequence_dict)
                if ref_gene and ref_gene not in used_indicator_genes:
                    used_indicator_genes.append(ref_gene)

        self.logic_genes = [g for g in self.logic_genes if g in used_logic_genes]
        self.indicator_genes = [
            g for g in self.indicator_genes if g in used_indicator_genes
        ]

        # Rebuild the genome with the filtered genes
        indicator_ast_list, logic_ast_list, algorithm_parameter_specs = (
            self._prepare_genes()
        )

        self.function_ast, self.param_space = self._construct_algorithm(
            indicator_ast_list=indicator_ast_list,
            logic_ast_list=logic_ast_list,
            algorithm_parameter_specs=algorithm_parameter_specs,
        )
        self.param_space = paramspecs_to_dict(self.param_space)
        self.compiled_function = ast_to_function(self.function_ast)
        self.compiled_function.param_space = self.param_space

        self.sequence_dict = {}
        for gene in self.indicator_genes + self.logic_genes + self.signal_genes:
            if isinstance(gene, IndicatorGene):
                self.sequence_dict[gene.get_name()] = gene  # Now single string
            else:
                self.sequence_dict[gene.get_name()] = gene

    def mutate(self) -> "Genome":
        current_genes = self.get_genes()
        selected_gene_idx = random.randint(0, len(current_genes) - 1)
        selected_gene = current_genes.pop(selected_gene_idx)

        if isinstance(selected_gene, IndicatorGene):
            mutated_gene = generate_indicator_gene()
        elif isinstance(selected_gene, LogicGene):
            current_logic_genes = [g for g in current_genes if isinstance(g, LogicGene)]
            mutated_gene = generate_logic_gene_sequence(
                num_logic=1,
                indicator_genes=self.indicator_genes,
                allow_logic_composition=True,
                logic_composition_prob=0.5,
                existing_logic_genes=(
                    current_logic_genes if current_logic_genes else None
                ),
            )[0]
        elif isinstance(selected_gene, SignalGene):
            mutated_gene = generate_signal_gene(self.logic_genes)

        current_genes[selected_gene_idx] = mutated_gene
        indicator_genes, logic_genes, signal_genes = separate_genes(current_genes)
        return Genome(
            indicator_genes=indicator_genes,
            logic_genes=logic_genes,
            signal_genes=signal_genes,
        )

    ### OPTIMIZATION METHODS
    def set_best(self, params: dict, metric: float) -> None:
        self._best_params = params
        self._best_metric = metric

    def get_best_params(self) -> dict | None:
        return self._best_params

    def get_best_metric(self) -> float | None:
        return self._best_metric

    def get_param_space(self) -> dict:
        return self.param_space

    ### GENE METHODS
    def get_num_indicators(self) -> int:
        return len(self.indicator_genes)

    def get_num_logic(self) -> int:
        return len(self.logic_genes)

    def get_num_signals(self) -> int:
        return len(self.signal_genes)

    def get_genes(self) -> tuple[IndicatorGene, LogicGene, SignalGene]:
        return self.indicator_genes + self.logic_genes + self.signal_genes

    def get_sequence_dict(self) -> dict:
        return self.sequence_dict

    ### FUNCTIONAL METHODS
    def get_ast(self) -> ast.AST:
        return self.function_ast

    def get_compiled_function(self) -> callable:
        return self.compiled_function

    def get_complexity(self, shannon_entropy: bool = False) -> int:
        if shannon_entropy:
            function_str = unparsify(self.function_ast)
            counts = Counter(function_str)
            total = len(function_str)
            entropy = 0
            for count in counts.values():
                p = count / total
                entropy += p * np.log2(p)
        else:
            return (
                len(self.indicator_genes)
                + len(self.logic_genes)
                + len(self.signal_genes)
            )

    def __call__(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return self.compiled_function(data, **kwargs)
