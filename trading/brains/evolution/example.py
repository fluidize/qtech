from genetics.ast_builder import generate_genome
from genetics.tools import display_ast

for x in range(100):
    genome = generation.generate_genome(logic_composition_prob=0.5)
    display_ast(genome.get_ast())
    print(genome.get_complexity())