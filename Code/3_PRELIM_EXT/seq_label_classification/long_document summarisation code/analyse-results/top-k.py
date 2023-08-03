import pandas as pd
import sys
import os, shutil

model = sys.argv[1]
data = sys.argv[2]

print('Analyzing results for {0} model and {1} data'.format(model, data))

if model == "gpt2-perplexity":
    if data == "bb":
        path = "../gpt2-perplexity/beige_books_perpexity"
    elif data == "minutes":
        path = "../gpt2-perplexity/minutes_perpexity"
    elif data == "amicus-facts":
        path = "../gpt2-perplexity/amicus_facts_perpexity"
    elif data == "amicus":
        path = "../gpt2-perplexity/amicus_perpexity"

    outputDir = os.path.join(path, "results")

    if os.path.exists(outputDir):
        filelist = [f for f in os.listdir(outputDir)]
        for f in filelist:
            os.remove(os.path.join(outputDir, f))
    else:
        os.makedirs(outputDir)

    for filename in os.listdir(path):
        if "scores" not in filename:
            continue
        print(filename)
        outfilename = "results_" + filename[7:]
        file = os.path.join(path, filename)
        outfile = os.path.join(outputDir, outfilename)

        scores = pd.read_csv(file)
        scores = scores.sort_values(['preplexity'], ascending=True).groupby('id').head(3)
        scores = scores.sort_values('id')
        scores.to_csv(outfile, index=False)


if model == "roberta-entailment":
    if data == "bb":
        path = "../roberta-entailment/beige_books_entailment"
    elif data == "minutes":
        path = "../roberta-entailment/minutes_entailment"
    elif data == "amicus-facts":
        path = "../roberta-entailment/amicus_facts_entailment"
    elif data == "amicus":
        path = "../roberta-entailment/amicus_entailment"

    outputDir = os.path.join(path, "results")
    if os.path.exists(outputDir):
        filelist = [f for f in os.listdir(outputDir)]
        for f in filelist:
            os.remove(os.path.join(outputDir, f))
    else:
        os.makedirs(outputDir)

    for filename in os.listdir(path):
        if "scores" not in filename:
            continue
        print(filename)
        outfilename = "results_" + filename[7:]
        file = os.path.join(path, filename)
        outfile = os.path.join(outputDir, outfilename)

        scores = pd.read_csv(file)
        scores = scores[scores['type'] == 2]
        scores = scores.sort_values(['prob'], ascending=False).groupby('id').head(3)
        scores = scores.sort_values('id')
        scores.to_csv(outfile, index=False)


if model == "baselines-bert":
    if data == "bb":
        path = "../baselines-bert/beige_books_bertemb"
    elif data == "minutes":
        path = "../baselines-bert/minutes_bertemb"
    elif data == "amicus-facts":
        path = "../baselines-bert/amicus_facts_bertemb"
    elif data == "amicus":
        path = "../baselines-bert/amicus_bertemb"

    outputDir = os.path.join(path, "results")
    if os.path.exists(outputDir):
        filelist = [f for f in os.listdir(outputDir)]
        for f in filelist:
            os.remove(os.path.join(outputDir, f))
    else:
        os.makedirs(outputDir)

    for filename in os.listdir(path):
        if "scores" not in filename:
            continue
        print(filename)
        outfilename = "results_" + filename[7:]
        file = os.path.join(path, filename)

        outfile = os.path.join(outputDir, outfilename)

        scores = pd.read_csv(file)
        scores = scores.sort_values(['similarity'], ascending=False).groupby('id').head(3)
        scores = scores.sort_values('id')
        scores.to_csv(outfile, index=False)


if model == "baselines-bleu":
    if data == "bb":
        path = "../baseline-bleu/beige_books_blue"
    elif data == "minutes":
        path = "../baseline-bleu/minutes_blue"
    elif data == "amicus-facts":
        path = "../baseline-bleu/amicus_facts_blue"
    elif data == "amicus":
        path = "../baseline-bleu/amicus_blue"

    outputDir = os.path.join(path, "results")
    if os.path.exists(outputDir):
        filelist = [f for f in os.listdir(outputDir)]
        for f in filelist:
            os.remove(os.path.join(outputDir, f))
    else:
        os.makedirs(outputDir)

    for filename in os.listdir(path):
        if "scores" not in filename:
            continue
        print(filename)
        outfilename = "results_" + filename[7:]
        file = os.path.join(path, filename)

        outfile = os.path.join(outputDir, outfilename)

        scores = pd.read_csv(file)
        scores = scores.sort_values(['Blue score'], ascending=False).groupby('id').head(3)
        scores = scores.sort_values('id')
        scores.to_csv(outfile, index=False)
