import ast
import evolution
from genetics.gp_tools import display_ast

individual = evolution.generate_individual()
display_ast(individual.get_function_ast())