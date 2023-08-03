import pandas
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

mlx = load_dataset('isebire/mlx_PEGASUS')
splits = ['train', 'validation', 'test']

for split in splits:
  filename = 'mlx_pegasus_' + split + '.csv'
  mlx[split].to_csv(filename)
