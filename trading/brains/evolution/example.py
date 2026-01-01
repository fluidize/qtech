import evolution
from genetics.ast_tools import display_ast
from evolution import mutate_genome

individual = evolution.generate_individual()
display_ast(individual.get_function_ast())
display_ast(mutate_genome(individual, p_mutate=1).get_function_ast())