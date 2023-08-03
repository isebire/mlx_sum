import pandas
import numpy as np

# Tiny versions of OREO datasets
def chunk(input_filename):
    df = pandas.read_json(input_filename, lines=True)
    df_chunks = np.array_split(df, 10)
    for i, chunk in enumerate(df_chunks):
        filename = 'chunks/CHUNK' + str(i+1) + input_filename
        chunk.to_json(filename, lines=True, orient='records')

chunk('train_mlx_bert.json')
chunk('validation_mlx_bert.json')
chunk('test_mlx_bert.json')
