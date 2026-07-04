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
        left_indicator_idx: int,
        right_indicator_idx: int,
        operator: ast.cmpop,
    ):
        self.left_indicator_idx = left_indicator_idx
        self.right_indicator_idx = right_indicator_idx
        self.operator = operator

    def load_indicators(self, indicator_variable_names: list[str]):
        left_name = indicator_variable_names[self.left_indicator_idx]
        right_name = indicator_variable_names[self.right_indicator_idx]
        return f"LOGIC_{left_name}_{type(self.operator).__name__}_{right_name}"

    def get_parameter_specs(self):
        return []

    def get_referenced_indicator_indices(self):
        return [self.left_indicator_idx, self.right_indicator_idx]

    def to_ast(self, indicator_variable_names: list[str]):
        left_name = indicator_variable_names[self.left_indicator_idx]
        right_name = indicator_variable_names[self.right_indicator_idx]
        variable_name = f"LOGIC_{left_name}_{type(self.operator).__name__}_{right_name}"
        compare_ast = make_compare(
            ast.Name(id=left_name, ctx=ast.Load()),
            self.operator,
            ast.Name(id=right_name, ctx=ast.Load()),
        )
        return ast.Assign(
            targets=[ast.Name(id=variable_name, ctx=ast.Store())],
            value=compare_ast,
        )

    def __eq__(self, other):
        if not isinstance(other, IndicatorToIndicator):
            return False
        return (
            self.left_indicator_idx == other.left_indicator_idx
            and self.right_indicator_idx == other.right_indicator_idx
            and type(self.operator) == type(other.operator)
        )

    def __hash__(self):
        return hash(
            (self.left_indicator_idx, self.right_indicator_idx, type(self.operator))
        )


# i2p
class IndicatorToPrice(LogicGene):
    columns = ["Open", "High", "Low", "Close"]

    def __init__(self, left_indicator_idx: int, column_index: int, operator: ast.cmpop):
        self.left_indicator_idx = left_indicator_idx
        self.column_index = column_index
        self.operator = operator

    def load_indicators(self, indicator_variable_names: list[str]):
        left_name = indicator_variable_names[self.left_indicator_idx]
        return f"LOGIC_{left_name}_{type(self.operator).__name__}_{self.columns[self.column_index]}"

    def get_parameter_specs(self):
        return []

    def get_referenced_indicator_indices(self):
        return [self.left_indicator_idx]

    def to_ast(self, indicator_variable_names: list[str]):
        left_name = indicator_variable_names[self.left_indicator_idx]
        variable_name = f"LOGIC_{left_name}_{type(self.operator).__name__}_{self.columns[self.column_index]}"
        compare_ast = make_compare(
            ast.Name(id=left_name, ctx=ast.Load()),
            self.operator,
            ast.Subscript(
                value=ast.Name(id="data", ctx=ast.Load()),
                slice=ast.Constant(value=self.columns[self.column_index]),
                ctx=ast.Load(),
            ),
        )
        return ast.Assign(
            targets=[ast.Name(id=variable_name, ctx=ast.Store())],
            value=compare_ast,
        )

    def __eq__(self, other):
        if not isinstance(other, IndicatorToPrice):
            return False
        return (
            self.left_indicator_idx == other.left_indicator_idx
            and self.column_index == other.column_index
            and type(self.operator) == type(other.operator)
        )

    def __hash__(self):
        return hash((self.left_indicator_idx, self.column_index, type(self.operator)))


# i2c
class IndicatorToConstant(LogicGene):
    def __init__(self, left_indicator_idx: int, operator: ast.cmpop):
        self.left_indicator_idx = left_indicator_idx
        self.operator = operator
        self.parameter_specs = None
        self._variable_name = None

    def load_indicators(self, indicator_variable_names: list[str]):
        left_name = indicator_variable_names[self.left_indicator_idx]
        counter = unique_counter()
        self._variable_name = (
            f"LOGIC_{left_name}_{type(self.operator).__name__}_const_{counter}"
        )
        self.parameter_specs = [
            ParamSpec(
                parameter_name=f"{self._variable_name}_constant",
                search_space=(-100, 100),
            )
        ]
        return self._variable_name

    def get_parameter_specs(self):
        return self.parameter_specs

    def get_referenced_indicator_indices(self):
        return [self.left_indicator_idx]

    def to_ast(self, indicator_variable_names: list[str]):
        left_name = indicator_variable_names[self.left_indicator_idx]
        if self._variable_name is None:
            # Fallback if load_indicators wasn't called
            counter = unique_counter()
            self._variable_name = (
                f"LOGIC_{left_name}_{type(self.operator).__name__}_const_{counter}"
            )
            self.parameter_specs = [
                ParamSpec(
                    parameter_name=f"{self._variable_name}_constant",
                    search_space=(-100, 100),
                )
            ]
        compare_ast = make_compare(
            ast.Name(id=left_name, ctx=ast.Load()),
            self.operator,
            ast.Name(id=f"{self._variable_name}_constant", ctx=ast.Load()),
        )
        return ast.Assign(
            targets=[ast.Name(id=self._variable_name, ctx=ast.Store())],
            value=compare_ast,
        )

    def __eq__(self, other):
        if not isinstance(other, IndicatorToConstant):
            return False
        return self.left_indicator_idx == other.left_indicator_idx and type(
            self.operator
        ) == type(other.operator)

    def __hash__(self):
        return hash((self.left_indicator_idx, type(self.operator)))


# l2l
class LogicToLogic(LogicGene):
    def __init__(self, left_logic_idx: int, right_logic_idx: int, operator: ast.cmpop):
        self.left_logic_idx = left_logic_idx
        self.right_logic_idx = right_logic_idx
        self.operator = operator
        self._variable_name = None

    def load_logic(self, logic_variable_names: list[str]):
        counter = unique_counter()
        self._variable_name = f"LOGIC_COMPOSITE_{counter}"
        return self._variable_name

    def get_parameter_specs(self):
        return []

    def get_referenced_logic_indices(self):
        return [self.left_logic_idx, self.right_logic_idx]

    def to_ast(self, logic_variable_names: list[str]):
        if self._variable_name is None:
            # Fallback if load_logic wasn't called
            counter = unique_counter()
            self._variable_name = f"LOGIC_COMPOSITE_{counter}"
        left_expr = ast.Name(
            id=logic_variable_names[self.left_logic_idx], ctx=ast.Load()
        )
        right_expr = ast.Name(
            id=logic_variable_names[self.right_logic_idx], ctx=ast.Load()
        )
        return ast.Assign(
            targets=[ast.Name(id=self._variable_name, ctx=ast.Store())],
            value=ast.BinOp(left=left_expr, op=self.operator, right=right_expr),
        )

    def __eq__(self, other):
        if not isinstance(other, LogicToLogic):
            return False
        return (
            self.left_logic_idx == other.left_logic_idx
            and self.right_logic_idx == other.right_logic_idx
            and type(self.operator) == type(other.operator)
        )

    def __hash__(self):
        return hash((self.left_logic_idx, self.right_logic_idx, type(self.operator)))


class SignalGene:
    def __init__(self, long_logic_idx: int, short_logic_idx: int):
        self.long_logic_idx = long_logic_idx
        self.short_logic_idx = short_logic_idx

    def get_referenced_logic_indices(self):
        return [self.long_logic_idx, self.short_logic_idx]

    def to_ast(self, logic_variable_names: list[str]):
        long_name = logic_variable_names[self.long_logic_idx]
        short_name = logic_variable_names[self.short_logic_idx]
        buy_conditions_assign = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id="signals", ctx=ast.Load()),
                    slice=ast.Name(id=long_name, ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=1),
        )
        sell_conditions_assign = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id="signals", ctx=ast.Load()),
                    slice=ast.Name(id=short_name, ctx=ast.Load()),
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
            self.long_logic_idx == other.long_logic_idx
            and self.short_logic_idx == other.short_logic_idx
        )

    def __hash__(self):
        return hash((self.long_logic_idx, self.short_logic_idx))


def _get_indicators_cached():
    global _INDICATORS_CACHE
    if _INDICATORS_CACHE is None:
        _INDICATORS_CACHE = get_indicators(exclude=EXCLUDED_INDICATORS)
    return _INDICATORS_CACHE


def generate_indicator_gene() -> IndicatorGene:
    return IndicatorGene(function=random.choice(_get_indicators_cached()))


def generate_logic_gene_sequence(
    num_logic: int,
    num_indicators: int,
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
            left_idx, right_idx = random.sample(range(len(existing_logic_genes)), 2)
            logic_genes.append(
                LogicToLogic(
                    left_logic_idx=left_idx,
                    right_logic_idx=right_idx,
                    operator=random_composition_operator(),
                )
            )
        else:
            # Choose logic type based on available indicators
            if num_indicators >= 2:
                logic_type = random.choice(
                    [IndicatorToPrice, IndicatorToConstant, IndicatorToIndicator]
                )
            else:
                # Only allow types that work with single indicator
                logic_type = random.choice([IndicatorToPrice, IndicatorToConstant])

            if logic_type == IndicatorToPrice:
                logic_genes.append(
                    IndicatorToPrice(
                        left_indicator_idx=random.randint(0, num_indicators - 1),
                        column_index=random.randint(0, 3),
                        operator=random_comparison_operator(),
                    )
                )
            elif logic_type == IndicatorToConstant:
                logic_genes.append(
                    IndicatorToConstant(
                        left_indicator_idx=random.randint(0, num_indicators - 1),
                        operator=random_comparison_operator(),
                    )
                )
            else:  # IndicatorToIndicator
                left_idx = random.randint(0, num_indicators - 1)
                right_idx = random.randint(0, num_indicators - 1)
                while right_idx == left_idx:
                    right_idx = random.randint(0, num_indicators - 1)
                logic_genes.append(
                    IndicatorToIndicator(
                        left_indicator_idx=left_idx,
                        right_indicator_idx=right_idx,
                        operator=random_comparison_operator(),
                    )
                )
    return logic_genes


def generate_signal_gene(num_logic: int) -> SignalGene:
    if num_logic < 2:
        raise ValueError("Need at least 2 logic genes to generate signal gene")
    long_idx, short_idx = random.sample(range(num_logic), 2)
    return SignalGene(long_logic_idx=long_idx, short_logic_idx=short_idx)


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

    num_logic = max(num_logic, 2)

    indicator_genes = [generate_indicator_gene() for _ in range(num_indicators)]

    logic_genes = generate_logic_gene_sequence(
        num_logic=num_logic,
        num_indicators=num_indicators,
        allow_logic_composition=allow_logic_composition,
        logic_composition_prob=logic_composition_prob,
    )
    signal_genes = [generate_signal_gene(num_logic=num_logic)]
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

        (
            indicator_ast_list,
            logic_ast_list,
            signal_ast_list,
            algorithm_parameter_specs,
        ) = self._prepare_genes()

        self.function_ast, self.param_space = self._construct_algorithm(
            indicator_ast_list=indicator_ast_list,
            logic_ast_list=logic_ast_list,
            signal_ast_list=signal_ast_list,
            algorithm_parameter_specs=algorithm_parameter_specs,
        )
        self.param_space = paramspecs_to_dict(self.param_space)
        self.compiled_function = ast_to_function(self.function_ast)
        self.compiled_function.param_space = self.param_space

        self._best_params = None
        self._best_metric = None

    def build_dependency_graph(self):
        graph = {"logic_to_indicators": {}, "signal_to_logic": {}, "logic_to_logic": {}}

        for i, logic_gene in enumerate(self.logic_genes):
            graph["logic_to_indicators"][i] = set(
                logic_gene.get_referenced_indicator_indices()
            )
            if isinstance(logic_gene, LogicToLogic):
                graph["logic_to_logic"][i] = set(
                    logic_gene.get_referenced_logic_indices()
                )
            else:
                graph["logic_to_logic"][i] = set()

        for i, signal_gene in enumerate(self.signal_genes):
            graph["signal_to_logic"][i] = set(
                signal_gene.get_referenced_logic_indices()
            )

        return graph

    def get_affected_logic_genes(self, indicator_idx: int) -> set[int]:
        if indicator_idx >= len(self.indicator_genes):
            return set()

        graph = self.build_dependency_graph()
        affected = set()
        for logic_idx, ind_refs in graph["logic_to_indicators"].items():
            if indicator_idx in ind_refs:
                affected.add(logic_idx)
        return affected

    def get_affected_signal_genes(self, logic_idx: int) -> set[int]:
        if logic_idx >= len(self.logic_genes):
            return set()

        graph = self.build_dependency_graph()
        affected = set()
        for signal_idx, logic_refs in graph["signal_to_logic"].items():
            if logic_idx in logic_refs:
                affected.add(signal_idx)
        return affected

    def rebuild(self) -> "Genome":
        """Rebuild the genome with current gene lists"""
        return Genome(
            indicator_genes=self.indicator_genes,
            logic_genes=self.logic_genes,
            signal_genes=self.signal_genes,
        )

    ### MUTATION OPERATIONS - big change / knockouts

    def replace_indicator(self, indicator_idx: int) -> "Genome":
        if not self.indicator_genes:
            return self

        # Replace the indicator
        self.indicator_genes[indicator_idx] = generate_indicator_gene()

        # Get affected logic genes
        affected_logic_indices = self.get_affected_logic_genes(indicator_idx)

        # Regenerate affected logic genes
        new_logic_genes = []
        for i, logic_gene in enumerate(self.logic_genes):
            if i in affected_logic_indices:
                # Regenerate this logic gene
                new_logic_genes.extend(
                    generate_logic_gene_sequence(
                        num_logic=1,
                        num_indicators=len(self.indicator_genes),
                        allow_logic_composition=False,
                        logic_composition_prob=0.0,
                    )
                )
            else:
                new_logic_genes.append(logic_gene)

        self.logic_genes = new_logic_genes

        # Check if any signal genes were affected by logic gene changes
        # Since we regenerated logic genes, indices may have changed
        # Safest to regenerate signal genes
        self.signal_genes = [generate_signal_gene(num_logic=len(self.logic_genes))]

        return self.rebuild()

    def replace_logic_operator(self, logic_idx: int) -> "Genome":
        #mutate op only
        if not self.logic_genes or logic_idx >= len(self.logic_genes):
            return self

        self.logic_genes[logic_idx].operator = random_comparison_operator()
        return self.rebuild()

    def replace_logic_gene(self, logic_idx: int) -> "Genome":
        if not self.logic_genes or logic_idx >= len(self.logic_genes):
            return self

        # Replace the logic gene
        new_logic_genes = generate_logic_gene_sequence(
            num_logic=1,
            num_indicators=len(self.indicator_genes),
            allow_logic_composition=False,
            logic_composition_prob=0.0,
        )
        self.logic_genes[logic_idx] = new_logic_genes[0]

        # Regenerate signal genes (they may have referenced this logic gene)
        self.signal_genes = [generate_signal_gene(num_logic=len(self.logic_genes))]

        return self.rebuild()

    def replace_signal_gene(self, signal_idx: int) -> "Genome":
        if not self.signal_genes or signal_idx >= len(self.signal_genes):
            return self

        self.signal_genes[signal_idx] = generate_signal_gene(
            num_logic=len(self.logic_genes)
        )
        return self.rebuild()

    ### INCREMENTAL MUTATION OPERATIONS - tiny change

    def swap_indicators(self, idx1: int, idx2: int) -> "Genome":
        """Swap two indicators"""
        if len(self.indicator_genes) < 2:
            return self
        if idx1 >= len(self.indicator_genes) or idx2 >= len(self.indicator_genes):
            return self

        self.indicator_genes[idx1], self.indicator_genes[idx2] = (
            self.indicator_genes[idx2],
            self.indicator_genes[idx1],
        )
        return self.rebuild()

    def add_indicator(self) -> "Genome":
        """Add a new indicator and regenerate affected logic"""
        new_indicator = generate_indicator_gene()
        self.indicator_genes.append(new_indicator)

        # Add a new logic gene that uses the new indicator
        new_logic_genes = generate_logic_gene_sequence(
            num_logic=1,
            num_indicators=len(self.indicator_genes),
            allow_logic_composition=False,
            logic_composition_prob=0.0,
        )
        self.logic_genes.extend(new_logic_genes)

        # Regenerate signal genes
        self.signal_genes = [generate_signal_gene(num_logic=len(self.logic_genes))]

        return self.rebuild()

    def remove_indicator(self, indicator_idx: int) -> "Genome":
        """Remove an indicator and regenerate affected logic"""
        if len(self.indicator_genes) <= 1:
            return self  # Keep at least one indicator
        if indicator_idx >= len(self.indicator_genes):
            return self

        # Get affected logic genes before removal
        affected_logic_indices = self.get_affected_logic_genes(indicator_idx)

        # Remove the indicator
        self.indicator_genes.pop(indicator_idx)

        # Filter logic genes: remove those that referenced the removed indicator
        # Update indices in remaining logic genes
        new_logic_genes = []
        for i, logic_gene in enumerate(self.logic_genes):
            if i in affected_logic_indices:
                # Skip this logic gene (it referenced the removed indicator)
                continue
            else:
                # Update indices in this logic gene (shift down if > removed_idx)
                updated_gene = self._update_logic_indices(logic_gene, indicator_idx, -1)
                new_logic_genes.append(updated_gene)

        self.logic_genes = new_logic_genes

        # Ensure we have at least 2 logic genes
        if len(self.logic_genes) < 2:
            self.logic_genes.extend(
                generate_logic_gene_sequence(
                    num_logic=2 - len(self.logic_genes),
                    num_indicators=len(self.indicator_genes),
                    allow_logic_composition=False,
                    logic_composition_prob=0.0,
                )
            )

        # Regenerate signal genes (since logic genes changed/indices shifted)
        self.signal_genes = [generate_signal_gene(num_logic=len(self.logic_genes))]

        return self.rebuild()

    def _update_logic_indices(
        self, logic_gene: LogicGene, changed_idx: int, delta: int
    ) -> LogicGene:

        def adjust_idx(idx):
            # When removing (delta = -1): shift indices > changed_idx down
            # When adding (delta = +1): shift indices >= changed_idx up
            if delta < 0:
                return idx + delta if idx > changed_idx else idx
            else:
                return idx + delta if idx >= changed_idx else idx

        if isinstance(logic_gene, IndicatorToIndicator):
            new_left = adjust_idx(logic_gene.left_indicator_idx)
            new_right = adjust_idx(logic_gene.right_indicator_idx)
            return IndicatorToIndicator(new_left, new_right, logic_gene.operator)
        elif isinstance(logic_gene, IndicatorToPrice):
            new_left = adjust_idx(logic_gene.left_indicator_idx)
            return IndicatorToPrice(
                new_left, logic_gene.column_index, logic_gene.operator
            )
        elif isinstance(logic_gene, IndicatorToConstant):
            new_left = adjust_idx(logic_gene.left_indicator_idx)
            return IndicatorToConstant(new_left, logic_gene.operator)
        else:
            return logic_gene

    def add_logic_gene(self) -> "Genome":
        new_logic_genes = generate_logic_gene_sequence(
            num_logic=1,
            num_indicators=len(self.indicator_genes),
            allow_logic_composition=False,
            logic_composition_prob=0.0,
        )
        self.logic_genes.extend(new_logic_genes)

        # Regenerate signal genes
        self.signal_genes = [generate_signal_gene(num_logic=len(self.logic_genes))]

        return self.rebuild()

    def remove_logic_gene(self, logic_idx: int) -> "Genome":
        """Remove a logic gene and update indices"""
        if len(self.logic_genes) <= 2:
            return self  # Keep at least 2 logic genes
        if logic_idx >= len(self.logic_genes):
            return self

        # Get affected signal genes before removal
        affected_signal_indices = self.get_affected_signal_genes(logic_idx)

        # Remove the logic gene
        self.logic_genes.pop(logic_idx)

        # Update indices in remaining logic genes (for LogicToLogic)
        new_logic_genes = []
        for i, logic_gene in enumerate(self.logic_genes):
            if isinstance(logic_gene, LogicToLogic):
                new_left = logic_gene.left_logic_idx + (
                    -1 if logic_gene.left_logic_idx > logic_idx else 0
                )
                new_right = logic_gene.right_logic_idx + (
                    -1 if logic_gene.right_logic_idx > logic_idx else 0
                )
                updated_gene = LogicToLogic(new_left, new_right, logic_gene.operator)
                new_logic_genes.append(updated_gene)
            else:
                new_logic_genes.append(logic_gene)

        self.logic_genes = new_logic_genes

        # Update indices in signal genes
        new_signal_genes = []
        for i, signal_gene in enumerate(self.signal_genes):
            if i in affected_signal_indices:
                # This signal gene referenced the removed logic gene, regenerate it
                new_signal_genes.append(
                    generate_signal_gene(num_logic=len(self.logic_genes))
                )
            else:
                # Update indices in this signal gene
                new_long = signal_gene.long_logic_idx + (
                    -1 if signal_gene.long_logic_idx > logic_idx else 0
                )
                new_short = signal_gene.short_logic_idx + (
                    -1 if signal_gene.short_logic_idx > logic_idx else 0
                )
                updated_signal = SignalGene(new_long, new_short)
                new_signal_genes.append(updated_signal)

        self.signal_genes = new_signal_genes

        return self.rebuild()

    def incremental_mutate(self) -> "Genome":
        """Make one small, incremental change"""
        operations = [
            ("change_operator", self._mutate_change_operator),
            ("replace_indicator", self._mutate_replace_indicator),
            ("replace_logic", self._mutate_replace_logic),
            ("replace_signal", self._mutate_replace_signal),
        ]

        # Add structural operations if we have enough genes
        if len(self.indicator_genes) >= 2:
            operations.append(("swap_indicators", self._mutate_swap_indicators))
        if len(self.indicator_genes) < 6:
            operations.append(("add_indicator", self._mutate_add_indicator))
        if len(self.indicator_genes) > 2:
            operations.append(("remove_indicator", self._mutate_remove_indicator))
        if len(self.logic_genes) < 6:
            operations.append(("add_logic", self._mutate_add_logic))
        if len(self.logic_genes) > 2:
            operations.append(("remove_logic", self._mutate_remove_logic))

        # Choose random operation
        operation_name, operation_func = random.choice(operations)
        return operation_func()

    def _mutate_change_operator(self) -> "Genome":
        """Change a random operator"""
        if not self.logic_genes:
            return self
        logic_idx = random.randint(0, len(self.logic_genes) - 1)
        return self.replace_logic_operator(logic_idx)

    def _mutate_replace_indicator(self) -> "Genome":
        """Replace a random indicator"""
        if not self.indicator_genes:
            return self
        indicator_idx = random.randint(0, len(self.indicator_genes) - 1)
        return self.replace_indicator(indicator_idx)

    def _mutate_replace_logic(self) -> "Genome":
        """Replace a random logic gene"""
        if not self.logic_genes:
            return self
        logic_idx = random.randint(0, len(self.logic_genes) - 1)
        return self.replace_logic_gene(logic_idx)

    def _mutate_replace_signal(self) -> "Genome":
        """Replace a random signal gene"""
        if not self.signal_genes:
            return self
        signal_idx = random.randint(0, len(self.signal_genes) - 1)
        return self.replace_signal_gene(signal_idx)

    def _mutate_swap_indicators(self) -> "Genome":
        """Swap two random indicators"""
        idx1, idx2 = random.sample(range(len(self.indicator_genes)), 2)
        return self.swap_indicators(idx1, idx2)

    def _mutate_add_indicator(self) -> "Genome":
        """Add a new indicator"""
        return self.add_indicator()

    def _mutate_remove_indicator(self) -> "Genome":
        """Remove a random indicator"""
        indicator_idx = random.randint(0, len(self.indicator_genes) - 1)
        return self.remove_indicator(indicator_idx)

    def _mutate_add_logic(self) -> "Genome":
        """Add a new logic gene"""
        return self.add_logic_gene()

    def _mutate_remove_logic(self) -> "Genome":
        """Remove a random logic gene"""
        logic_idx = random.randint(0, len(self.logic_genes) - 1)
        return self.remove_logic_gene(logic_idx)

    def random_replace_mutation(self) -> "Genome":
        """Uniformly random selection from all replace functions"""
        replace_operations = [
            ('replace_indicator', self._mutate_replace_indicator),
            ('replace_logic_operator', self._mutate_change_operator),
            ('replace_logic_gene', self._mutate_replace_logic),
            ('replace_signal_gene', self._mutate_replace_signal),
        ]

        operation_name, operation_func = random.choice(replace_operations)
        return operation_func()

    def random_incremental_mutation(self) -> "Genome":
        """Uniformly random selection from all incremental mutation functions"""
        incremental_operations = [
            ('swap_indicators', self._mutate_swap_indicators),
            ('add_indicator', self._mutate_add_indicator),
            ('remove_indicator', self._mutate_remove_indicator),
            ('add_logic', self._mutate_add_logic),
            ('remove_logic', self._mutate_remove_logic),
        ]

        # Filter operations based on current genome state
        available_operations = []
        if len(self.indicator_genes) >= 2:
            available_operations.append(('swap_indicators', self._mutate_swap_indicators))
        if len(self.indicator_genes) < 6:
            available_operations.append(('add_indicator', self._mutate_add_indicator))
        if len(self.indicator_genes) > 2:
            available_operations.append(('remove_indicator', self._mutate_remove_indicator))
        if len(self.logic_genes) < 6:
            available_operations.append(('add_logic', self._mutate_add_logic))
        if len(self.logic_genes) > 2:
            available_operations.append(('remove_logic', self._mutate_remove_logic))

        if not available_operations:
            # Fallback to replace mutations if no incremental operations available
            return self.random_replace_mutation()

        operation_name, operation_func = random.choice(available_operations)
        return operation_func()

    def mutate(self) -> "Genome":
        if random.random() < 0.5:
            return self.random_replace_mutation()
        else:
            return self.random_incremental_mutation()

    def _prepare_genes(self):
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
                variable_name = logic_gene.load_logic(logic_variable_names)
                logic_ast_list.append(logic_gene.to_ast(logic_variable_names))
            else:
                variable_name = logic_gene.load_indicators(indicator_variable_names)
                logic_ast_list.append(logic_gene.to_ast(indicator_variable_names))
            logic_variable_names.append(variable_name)
            algorithm_parameter_specs.extend(logic_gene.get_parameter_specs())

        signal_ast_list = []
        for signal_gene in self.signal_genes:
            signal_ast_list.extend(signal_gene.to_ast(logic_variable_names))

        return (
            indicator_ast_list,
            logic_ast_list,
            signal_ast_list,
            algorithm_parameter_specs,
        )

    def _construct_algorithm(
        self,
        indicator_ast_list,
        logic_ast_list,
        signal_ast_list,
        algorithm_parameter_specs,
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

        # return signals
        return_signals = ast.Return(value=ast.Name(id="signals", ctx=ast.Load()))

        body = [
            signals_init,
            *indicator_ast_list,
            *logic_ast_list,
            *signal_ast_list,
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
        # Get used logic indices from signals
        used_logic_indices = set()
        for signal_gene in self.signal_genes:
            used_logic_indices.update(signal_gene.get_referenced_logic_indices())

        # Find all logic genes that are used (directly or indirectly)
        used_logic_indices_set = set(used_logic_indices)
        i = 0
        while i < len(used_logic_indices_set):
            logic_idx = list(used_logic_indices_set)[i]
            if logic_idx < len(self.logic_genes):
                gene = self.logic_genes[logic_idx]
                if isinstance(gene, LogicToLogic):
                    for ref_idx in gene.get_referenced_logic_indices():
                        if ref_idx not in used_logic_indices_set:
                            used_logic_indices_set.add(ref_idx)
            i += 1

        # Get used indicator indices from used logic genes
        used_indicator_indices = set()
        for logic_idx in used_logic_indices_set:
            if logic_idx < len(self.logic_genes):
                gene = self.logic_genes[logic_idx]
                if isinstance(gene, IndicatorToIndicator):
                    used_indicator_indices.update(
                        gene.get_referenced_indicator_indices()
                    )
                elif isinstance(gene, IndicatorToPrice) or isinstance(
                    gene, IndicatorToConstant
                ):
                    used_indicator_indices.update(
                        gene.get_referenced_indicator_indices()
                    )

        self.logic_genes = [
            g for i, g in enumerate(self.logic_genes) if i in used_logic_indices_set
        ]
        self.indicator_genes = [
            g for i, g in enumerate(self.indicator_genes) if i in used_indicator_indices
        ]

        return Genome(
            indicator_genes=self.indicator_genes,
            logic_genes=self.logic_genes,
            signal_genes=self.signal_genes,
        )

    def mutate(self) -> "Genome":
        """Randomly choose between replace and incremental mutations"""
        mutation_type = random.choice(['replace', 'incremental'])
        if mutation_type == 'replace':
            return self.random_replace_mutation()
        else:
            return self.random_incremental_mutation()

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
