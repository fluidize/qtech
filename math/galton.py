import numpy as np
import matplotlib.pyplot as plt

def galton_board(rows, balls):
    bins = np.zeros(rows + 1)

    for _ in range(balls):
        position = 0
        for _ in range(rows):
            position += np.random.choice([0, 1])  # Move right (1) or stay (0)
        bins[position] += 1  # Increment bin count for the final position

    plt.bar(range(rows + 1), bins, color='blue', alpha=0.7)
    plt.xlabel('Bins (Final Position)')
    plt.ylabel('Number of Balls')
    plt.grid(True)
    plt.show()

rows = 10
balls = 1000
galton_board(rows, balls)