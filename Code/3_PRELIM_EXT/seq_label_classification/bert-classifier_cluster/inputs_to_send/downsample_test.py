import pandas

colnames = ['label', 'context']
df = pandas.read_csv('sample.tsv', sep='\t', names=colnames)

input(df)

df_positive = df[df['label'] == 1]
input(df_positive)
df_negative = df[df['label'] == 0]
input(df_negative)
# downsample negative dataset
df_positive_downsampled = df_positive.sample(df_negative.shape[0])
input(df_positive_downsampled)
df = pandas.concat([df_positive_downsampled, df_negative])
input(df)
# shuffle
df = df.sample(frac=1)

print(df)
