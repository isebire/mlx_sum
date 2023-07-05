import pandas

colnames = ['label', 'context']
df = pandas.read_csv('sample.tsv', sep='\t', names=colnames)
print(df.shape)
print(df['label'].value_counts())
