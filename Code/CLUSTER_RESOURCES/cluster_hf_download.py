# huggingface stuff needs to be downloaded for running on the cluster
from datasets import load_dataset
import pandas
from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "isebire"
FILENAME = "sklearn_model.joblib"

model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)

# Downloading a dataset to csv
mlx = load_dataset('isebire/mlx_CLEAN_FINAL'))
for split, data in mlx.items():
    data.to_csv(f"mlx-final-{split}.csv", index=None)

# Loading a dataset from csv
mlx = DatasetDict()
for split in ['train', 'validation', 'test']:
    split_df = pandas.read_csv(f"mlx-final-{split}.csv")
    split_ds = Dataset.from_pandas(split_df)
    mlx[split] = split_ds

# Downloading a model
# Do from cmd line
# git clone https://huggingface.co/nsi319/legal-pegasus

# Loading a model
# in the code: .from_pretrained(path to model)

# Saving a model
# trainer.save_model(path to save)
