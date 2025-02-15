import pandas as pd
df = pd.Series([1,10,100,1000,10000,100000])
print(df.pct_change())