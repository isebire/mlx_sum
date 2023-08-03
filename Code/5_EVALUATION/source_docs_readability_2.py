# Readability of source documents in test set :)

from datasets import load_dataset, load_from_disk
import statistics
import pandas
from readability_score.calculators.fleschkincaid import *
from readability_score.calculators.colemanliau import *
from readability_score.calculators.ari import *
from readability_score.calculators.smog import *

def remove_nones(data):
    return [i for i in data if i is not None]

def readability(text):
    metrics_dict = {}

    metrics_dict['flesch_kincaid'] = FleschKincaid(text).min_age
    metrics_dict['coleman_liau'] = ColemanLiau(text).min_age
    metrics_dict['ari'] = ARI(text).min_age
    metrics_dict['smog'] = SMOG(text).min_age

    return metrics_dict

print('************ NOW FOR OG')

print('*** Load dataset')
mlx = load_dataset("allenai/multi_lexsum", name="v20220616")
mlx_test = mlx['test']
print(len(mlx_test))

fk = []
fk2 = []
cl = []
ari = []
smog = []

for i, case in enumerate(mlx_test):
    print('** ON case ' + str(i))

    source_documents = case['sources']
    for doc in source_documents:
        metrics_dict = readability(doc)
        fk.append(metrics_dict['flesch_kincaid'])
        cl.append(metrics_dict['coleman_liau'])
        ari.append(metrics_dict['ari'])
        smog.append(metrics_dict['smog'])


fk = remove_nones(fk)
cl = remove_nones(cl)
ari = remove_nones(ari)
smog = remove_nones(smog)

print('** Mean time!!')
print('Flesch Kincaid')
print(statistics.mean(fk))
print('Coleman Liau')
print(statistics.mean(cl))
print('ARI')
print(statistics.mean(ari))
print('SMOG')
print(statistics.mean(smog))
