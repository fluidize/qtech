import evolution
from genetics.gp_tools import display_ast

for x in range(100):
    individual = evolution.generate_individual(min_indicators=2, max_indicators=10, min_logic=2, max_logic=10, logic_composition_prob=1)