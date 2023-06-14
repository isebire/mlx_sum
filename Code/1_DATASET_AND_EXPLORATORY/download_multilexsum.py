# downloading multi lex sum

from datasets import load_dataset

multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20220616")

example = multi_lexsum["validation"][0]
example["sources"]

for sum_len in ["long", "short", "tiny"]:
    print(example["summary/" + sum_len])

# Saving the dataset to csv

for split, data in multi_lexsum.items():
    data.to_csv(f"multilexsum-{split}.csv", index = None)
