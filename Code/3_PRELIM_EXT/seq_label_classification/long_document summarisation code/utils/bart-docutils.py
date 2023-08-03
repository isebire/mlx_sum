import os
import pandas as pd
import docutils
import pdb
import modelutils
import numpy as np 
import csv
config = {
    'dataType': {
        'amicus': 1,
        'bb': 2,
        'minutes': 3,
        'arxiv':4,
        'pubmed': 5
    }
}

def SevenNumberSummary(arr):
    # min, q1, median,q3, max, mean, stddev
    q1, med, q3 = np.percentile(arr, [25,50,75])
    print(f'Min: {np.min(arr)}, Q1: {q1}, Median: {med}, Q3: {q3}, Max: {np.max(arr)}, Mean: {np.mean(arr)}, Stddev: {np.std(arr)}')
    return [np.min(arr), q1, med, q3, np.max(arr), np.mean(arr), np.std(arr)]
def printUniqueTestFileIds(srcDir):
    #Func to print test split's fileIds
    entries = os.listdir(srcDir)
    def func(x):
        if x.split('.')[-1]=='csv' and x.split('_')[0]=='test' and x.split('_')[2]=='amicus':
            return True
        return False
    entries = list(filter(func, entries))
    for e in entries:
        print(e)
        df = pd.read_csv(os.path.join(srcDir,e))
        unq = df.fileId.unique()
        print(len(unq), unq)

def generateTestSrcCSV(srcDir, outDir, splitDir, dataType='amicus'):
    """
    Description: Take the test-split mapping of amicus, read corresponding sentence split src file for each file, 
    run invalid function, generate fileId, datatypeId, contextId, context csvs.

    srcDir: Where the 0.src etc are present
    outDir: where we want to write the final output file
    splitDir: Where we can read the test split information
    """
    test = np.loadtxt(os.path.join(splitDir, dataType+'_test_split.npy'))
    fout = open(os.path.join(outDir, dataType+ '_test_full.csv'), 'w')
    wout = csv.writer(fout)
    wout.writerow(['dataTypeId', 'fileId','context', 'context_id', 'label'])
    for entry in test:
        print(entry)
        sentences = open(os.path.join(srcDir, str(int(entry))+".src"), "r").read().split("\n")
        for idx, s in enumerate(sentences):
            s = modelutils.preprocessSentence(s)
            if len(s.split())<10 or len(s.strip())<5:
                continue
            wout.writerow([config['dataType'][dataType], str(entry), s, idx, 'unk'])
    fout.close()

def generateTestSrcFilesForInferring(srcDir, outDir, splitDir, filename, dataType='amicus'):
    """
    use the classifier output test files to generate extractive summary source file for bart.

    srcDir: Where classifier outputs are dumped
    outDir: Where to write the output files to.
    splitDir: Where we can get the test mapping for the dataset
    filename: What does the filename of the classifier output start with
    """
    test = np.loadtxt(os.path.join(splitDir, dataType+'_test_split.npy'))
    method = ["Blue", "preplexity", "prob", "similarity"]
    #sampling = ["agg", "topk"]
    sampling = ["random", "agg", "topk"]
    for m in method:
        for s in sampling:
            try:
                df = pd.read_csv(os.path.join(srcDir,f'{dataType}-{m}-{s}', filename+'_output.csv'))
                fsrc = open(os.path.join(outDir, f'{dataType}-{m}-{s}.src'), "w")
                summaries = []
                for t in test:
                    print(m,s,t)
                    df_id = df[df['fileId']==t]
                    df_id = df_id[df_id['predicted']==1]
                    df_id = df_id.sort_values(by='context_id')
                    context = df_id['context']
                    summary = " ".join(context)
                    summary= summary.replace(" ##", "")
                    summaries.append(summary)
                    #break
                fsrc.write("\n".join(summaries)+"\n")
                fsrc.close()
            except:
                f = os.path.join(srcDir,f'{dataType}-{m}-{s}', filename+'_output.csv')
                print(f'Error processing {f}')

def getCompressionRates(srcDir, docDir, splitDir, filename, dataType='amicus'):
    """
    Use the classifier output files and original sentence dumps to compute compression rates.

    srcDir: Where classifier outputs are dumped.
    docDir: Where original test documents sentence level splits are present
    splitDir: Where we can get the test mapping for the dataset
    filename: What does the filename of the classifier output start with
    """
    fout = open(os.path.join('.', dataType+ '_compression_stats.csv'), 'w')
    wout = csv.writer(fout)
    wout.writerow(['datatype','method', 'sampling','ratio_type', 'min', 'q1', 'median', 'q3', 'max', 'mean', 'stddev'])
    test = np.loadtxt(os.path.join(splitDir, dataType+'_test_split.npy'))
    method = ["Blue", "preplexity", "prob", "similarity"]
    sampling = ["agg", "topk", "random"]
    originalDocLength = []
    filteredDocLength = []

    for t in test:
        f = open(os.path.join(docDir, str(int(t))+".src"),"r")
        txt = f.read().split('\n')
        originalDocLength.append(len(txt))
        count = 0
        for idx, s in enumerate(txt):
            s = modelutils.preprocessSentence(s)
            if len(s.split())<10 or len(s.strip())<5:
                continue
            count+=1
        filteredDocLength.append(count)
        f.close()
    originalDocLength = np.array(originalDocLength)
    filteredDocLength = np.array(filteredDocLength)
    assert originalDocLength.shape == filteredDocLength.shape
    for m in method:
        for s in sampling:
            df = pd.read_csv(os.path.join(srcDir,f'amicus-{m}-{s}', filename+'_output.csv'))
            summaries = []
            words = []
            extractiveLength = []
            print(f'Method {m} sampling {s}')
            for t in test:
                #breakpoint()
                df_id = df[df['fileId']==t]
                df_id = df_id[df_id['predicted']==1]
                df_id = df_id.sort_values(by='context_id')
                context = df_id['context']
                extractiveLength.append(context.shape[0])
                words.append(len(" ".join(context).split()))
                #break
            extractiveLength = np.array(extractiveLength)
            assert extractiveLength.shape == originalDocLength.shape
            ratio_original_extractive = np.divide(extractiveLength,originalDocLength)
            s1 = SevenNumberSummary(ratio_original_extractive)
            wout.writerow([dataType, m, s, 'ratio_extractive_original']+s1)
            ratio_filtered_extractive = np.divide(extractiveLength,filteredDocLength)
            s2 = SevenNumberSummary(ratio_filtered_extractive)
            #breakpoint()
            wout.writerow([dataType, m, s, 'ratio_extractive_filtered']+s2)
            s3 = SevenNumberSummary(np.array(words))
            wout.writerow([dataType, m, s, 'words']+s3)
            print('Ratio_original_extractive', s1 )
            print('ratio_filtered_extractive', s2 )
            #break
        #break
    fout.close()
    return

def getOriginallStats(srcDir):
    """
    srcDir = Where sentence level splits are for 0.src, 0.tgt etc
    Want to get dataset characteristics.
    """
    sourceSent, targetSent = [], []
    sourceWords, targetWords = [], []
    fileEntries = os.listdir(srcDir)
    uniqueNames = set(map(lambda x: x.split(".")[0], fileEntries))
    for name in uniqueNames:
        fsrc = open(f'{srcDir}/{name}.src', "r")
        ftgt = open(f'{srcDir}/{name}.tgt', "r")
        source = fsrc.read().split("\n")
        target = ftgt.read().split("\n")
        sourceSent.append(len(source))
        targetSent.append(len(target))
        sw, tw = 0, 0
        for l in source:
            sw+=len(l.split())
        for l in target:
            tw+=len(l.split())
        sourceWords.append(sw)
        targetWords.append(tw)
        fsrc.close()
        ftgt.close()
    print("Source sentences")
    SevenNumberSummary(sourceSent)
    print("Source Words")
    SevenNumberSummary(sourceWords)
    print("Target Sentences")
    SevenNumberSummary(targetSent)
    print("Target words")
    SevenNumberSummary(targetWords)
    return


def generateRandomSrcForTest(docDir, outDir, splitDir, seed, classifierDir, filename, dataType='amicus'):
    """
    generate random compressed source comparable to classifier model outputs.

    docDir: Where original test documents sentence level splits are present
    outDir: Where to write the output files to.
    splitDir: Where we can get the test mapping for the dataset
    seed: seed for random sampling
    classifierDir: To read classifier output to determine length of random compressed source.
    filename: What does the filename of the classifier output start with
    """
    test = np.loadtxt(os.path.join(splitDir, dataType+'_test_split.npy'))
    method = ["Blue", "preplexity", "prob", "similarity"]
    sampling = ["agg", "topk", "random"]
    test_df = {}
    np.random.seed(seed)
    for t in test:
        dict_ = {
            'context_id': [],
            'context': []
        }
        f = open(os.path.join(docDir, str(int(t))+".src"),"r")
        txt = f.read().split('\n')
        for idx, s in enumerate(txt):
            s = modelutils.preprocessSentence(s)
            if len(s.split())<10 or len(s.strip())<5:
                continue
            dict_['context_id'].append(idx)
            dict_['context'].append(s) 
            test_df[t] = pd.DataFrame(dict_)
    for m in method:
        for s in sampling:
                try:
                    classifier_df = pd.read_csv(os.path.join(classifierDir, f'{dataType}-{m}-{s}', filename+'_output.csv'))
                    fsrc = open(os.path.join(outDir, f'{dataType}-{m}-{s}.src'), "w")
                    sources = []
                    #breakpoint()
                    for t in test:
                        df = test_df[t]
                        classifier_df['fileId'] = classifier_df['fileId'].astype(int)
                        classifier_df['predicted'] = classifier_df['predicted'].astype(int)
                        calssifier_t = classifier_df[classifier_df['fileId']==int(t)]
                        length = len(calssifier_t[calssifier_t['predicted']==1])
                        print(f'{dataType},{m},{s},{t}: {length}')
                        l = list(df['context_id'])
                        if len(l)>length:
                            samples = list(np.random.choice(l, length, replace = False))
                            df = df[df['context_id'].isin(samples)]
                        df = df.sort_values(by='context_id')
                        print(f'df: {df.shape[0]}')
                        source = ".".join(df['context'])
                        source = source.replace('\n', '').replace('\r', '')
                        sources.append(source)
                    fsrc.write("\n".join(sources)+"\n")
                    fsrc.close()
                except:
                    f = os.path.join(outDir, f'{dataType}-{m}-{s}.src')
                    print(f'Failed processing {f}')
    return