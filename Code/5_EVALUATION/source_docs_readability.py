# Readability of source documents in test set :)

from datasets import load_dataset, load_from_disk
import statistics
from readability import Readability
import pandas
from readability_score.calculators.fleschkincaid import *


def remove_nones(data):
    return [i for i in data if i is not None]

def readability(text):
    r = Readability(text)   # must contain at least 100 words
    metrics_dict = {}

    try:
        metrics_dict['flesch_kincaid'] = r.flesch_kincaid().score
    except:
        metrics_dict['flesch_kincaid'] = None

    try:
        metrics_dict['coleman_liau'] = r.coleman_liau().score
    except:
        metrics_dict['coleman_liau'] = None

    try:
        metrics_dict['ari'] = r.ari().score
    except:
        metrics_dict['ari'] = None

    try:
        metrics_dict['smog'] = r.smog().score
    except:
        metrics_dict['smog'] = None

    return metrics_dict

print('*** Load dataset')
mlx = load_dataset("isebire/mlx_CLEANED_ORDERED_CHAINS")
mlx_test = mlx['test']
print(len(mlx_test))

fk = []
fk2 = []
cl = []
ari = []
smog = []

for i, case in enumerate(mlx_test):
    print('** ON case ' + str(i))

    source_documents = case['sources_clean']
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
