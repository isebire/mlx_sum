import pandas

# Tiny versions of OREO datasets
df = pandas.read_json('train_mlx_bert.json', lines=True)
df_small = df.head(14)
df_small.to_json('TINY_train_mlx_bert.json', lines=True, orient='records')

df = pandas.read_json('validation_mlx_bert.json', lines=True)
df_small = df.head(2)
df_small.to_json('TINY_validation_mlx_bert.json', lines=True, orient='records')

df = pandas.read_json('train_mlx_bert.json', lines=True)
df_small = df.head(4)
df_small.to_json('TINY_test_mlx_bert.json', lines=True, orient='records')
