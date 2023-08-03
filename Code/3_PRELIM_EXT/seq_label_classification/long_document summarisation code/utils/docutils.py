import os
from utils.modelutils import initializeModel, processPairGPT2, processPairRoberta, \
    processPairBERTEmb, processPairNGRAM, processPairBLEU
import pandas as pd
import numpy as np
def retreiveSrcTgtInner(srcDir, entryList, formatType=''):
    pairs = []
    for name in entryList:
        if name == '':  # if filenames don't start with 0.src
            continue
        try:
            fsource = open(os.path.join(srcDir, str(name)+".src"),"r")
            ftarget = open(os.path.join(srcDir, str(name)+".tgt"), "r")
        except:
            print('Couldnt find', name)
            continue
        source = fsource.read()
        target = ftarget.read()
        if formatType == 'sentences':
            source = "".join(source)
            target = "".join(target)
            source = source.replace('\n', '').replace('\r', '')
            target = target.replace('\n', '').replace('\r', '')
        pairs = pairs + [(name, source,target)]
        fsource.close()
        ftarget.close()
    return pairs

def retreiveSrcTgtPairs(srcDir, numSamples, formatType=''):
    fileEntries = os.listdir(srcDir)
    uniqueNames = set(map(lambda x: x.split(".")[0], fileEntries))
    uniqueNames = sorted(list(uniqueNames))
    if numSamples != 0:  # default value = 0
        uniqueNames = uniqueNames[:numSamples]
    else:
        numSamples = len(uniqueNames)
    print('Computing for {0} document pairs'.format(numSamples))
    return retreiveSrcTgtInner(srcDir, uniqueNames, formatType=formatType)

def retreiveSrcTgtPairsGivenEntries(srcDir, entryList, formatType=''):
    return retreiveSrcTgtInner(srcDir, entryList, formatType=formatType)

def processData(args, model_type):
    numSamples = args.length  # default value = 0 (use all)
    numInputs = args.numInputs
    print('Considering {0} input sentences together as context'.format(numInputs))
    pairs = retreiveSrcTgtPairs(args.sourceDir, numSamples)
    print('Retrieved Document pairs: ', len(pairs))
    model, tokenizer = initializeModel(model_type=model_type, gpu=args.gpu)
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
    for idx, entry in enumerate(pairs):
        print('Processing Doc Pair..', idx + 1, 'with name', entry[0])
        if model_type == "gpt2":
            processPairGPT2((entry[1],entry[2]), model, tokenizer, args.gpu, args.outputDir, entry[0], numInputs)
        elif model_type == "roberta":
            processPairRoberta((entry[1],entry[2]), model, tokenizer, args.gpu, args.outputDir, entry[0], numInputs)
        elif model_type == "baselines-bert":
            processPairBERTEmb((entry[1],entry[2]), model, tokenizer, args.gpu, args.outputDir, entry[0], numInputs)
        elif model_type == "baselines-ngram":
            processPairNGRAM((entry[1],entry[2]), model, tokenizer, args.gpu, args.outputDir, entry[0], numInputs)
        elif model_type == "baselines-bleu":
            processPairBLEU((entry[1],entry[2]), model, tokenizer, args.gpu, args.outputDir, entry[0], numInputs)
    print("Done with computation!")



def combineAllSrcTgt(sourceDir, outputDir, filename, formatType='',readSplit=False, splitType='', dataType=''):
    print(formatType)
    #If a folder contains 0.src,0.tgt,1.src,1.tgt etc..., this generates combined .src .tgt files that has each example seperated by a line.
    pairs = None
    if not readSplit:
        pairs = retreiveSrcTgtPairs(sourceDir, 0, formatType=formatType)
    else:
        entries = test = np.loadtxt(os.path.join(splitDir, dataType+'_'+splitType+'_split.npy')).astype(int)
        pairs = retreiveSrcTgtPairsGivenEntries(sourceDir, entries, formatType=formatType)
    fsource = open(os.path.join(outputDir, filename+".src"),"w")
    ftarget = open(os.path.join(outputDir, filename+".tgt"), "w")
    names, source, target = list(zip(*pairs))
    print(len(source), len(target))
    fsource.write("\n".join(source)+"\n")
    ftarget.write("\n".join(target)+"\n")
    fsource.close()
    ftarget.close()
    print('Done')
    return


def convertPositiveSamplesToTargetSummaries(filename, outputDir, formatType='amicus', outFile = ''):
    samples = pd.read_csv(filename)
    unq = samples.fileId.unique()
    summaries = []
    unq = sorted(list(unq), key=lambda x: int(x))
    for entry in unq:
        df = samples[samples['fileId']==entry]
        df = df[df['label']=='positive']
        df = df.sort_values(by='context_id')
        context = df['context']
        summary = " ".join(context)
        if formatType == 'amicus':
            ftgt = open(os.path.join(outputDir, str(entry)+".src"), "w")
            ftgt.write(summary)
            ftgt.close()
        else:
            summaries.append(summary)
    if len(summaries)>0:
        ftarget = open(os.path.join(outputDir, outFile), "w")
        ftarget.write("\n".join(summaries)+"\n")
        ftarget.close()
    print('done')




