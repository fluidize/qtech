import individual
from genetics.ast_tools import display_ast

individual = individual.generate_individual()
display_ast(individual.construct_algorithm()[0])