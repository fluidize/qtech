import os
import pandas as pd
folder_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder_dir)
combined = pd.DataFrame()
for csv in os.listdir(folder_dir):
    if csv != "combine.py":
        combined = pd.concat([combined, pd.read_csv(csv)])
combined.to_csv("combined.csv")