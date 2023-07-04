import pandas

# Tiny versions of PEGASUS datasets
df = pandas.read_csv('mlx-final-train.csv')
df_small = df.head(14)
df_small.to_csv('TINY-mlx-final-train.csv')

df = pandas.read_csv('mlx-final-validation.csv')
df_small = df.head(2)
df_small.to_csv('TINY-mlx-final-validation.csv')

df = pandas.read_csv('mlx-final-test.csv')
df_small = df.head(4)
df_small.to_csv('TINY-mlx-final-test.csv')
