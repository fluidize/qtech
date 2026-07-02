### Evolve a single lineage

from trading.backtesting.backtesting import VectorizedBacktest
import trading.backtesting.mc_analysis as mc
import trading.model_tools as mt

from genetics.ast_builder import generate_population, generate_genome
from genetics.tools import display_ast, unparsify

import ast
from tqdm import tqdm
from rich import print
from time import time
import matplotlib.pyplot as plt
import numpy as np
import os

import faulthandler
faulthandler.enable()

founder = generate_genome(
    num_indicators=2,
    num_logic=2,
    allow_logic_composition=False
) #the founder is very simple

print(founder.mutate())