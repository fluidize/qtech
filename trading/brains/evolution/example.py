import evolution
from genetics.gp_tools import display_ast

for x in range(100):
    individual = evolution.generate_individual(logic_composition_prob=1)
    display_ast(individual.get_function_ast())
    individual.remove_unused_genes()
    display_ast(individual.get_function_ast())