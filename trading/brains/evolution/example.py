from genetics.algorithm_builder import generate_genome
from genetics.tools import display_ast


genome = generate_genome(logic_composition_prob=0.5)
display_ast(genome.get_ast())
print("Complexity: ", genome.get_complexity())