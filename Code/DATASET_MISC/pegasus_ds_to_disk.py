import pandas
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

mlx = load_dataset('isebire/mlx_PEGASUS')
splits = ['train', 'validation', 'test']

for split in splits:

  print(len(mlx[split]))
  filename = 'mlx_PEGASUS_' + split + '.hf'
  ds = mlx[split]
  ds.save_to_disk(filename)
