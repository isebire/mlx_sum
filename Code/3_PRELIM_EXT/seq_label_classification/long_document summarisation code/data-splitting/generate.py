import csv
import pandas as pd 
import argparse
import os
import pdb
from statistics import mean 
import numpy as np
np.random.seed(123)
config = {
    'dataType': {
        'amicus': 1,
        'bb': 2,
        'minutes': 3,
        'arxiv':4,
        'pubmed': 5,
        'newsroom': 6
    }
}

def getFileEntries(srcDir, numInputs):
    fileEntries = os.listdir(srcDir)
    def fun(x):
        if x.split('_')[-1] == str(numInputs)+".csv" and x.split('_')[0]=='scores':
           return True
        return False
    return filter(fun, fileEntries)

def splitUsingTopK(args):
    print(args)
    entries = getFileEntries(args.sourceDir, args.numInputs)
    fsamples = open(os.path.join(args.outputDir, 'samples_topk_' + args.dataType +"_"+ args.field.split()[0]+ '.csv'), 'w')
    wcsample = csv.writer(fsamples)
    wcsample.writerow(['dataTypeId', 'fileId','context', 'context_id', 'label'])
    top,bottom = None, None
    for entry in entries:
        fileId = entry.split('_')[-2]
        df = pd.read_csv(os.path.join(args.sourceDir, entry))
        unq = df.id.unique()
        k = args.k
        while 2*k*len(unq) > len(df):
            k = k-1
        for val in unq:
            df_id = df[df['id']==val]
            top_id = df_id.nlargest(2*k, args.field, keep='all')
            bottom_id = df_id.nsmallest(2*k, args.field, keep='all')
            if top is None:
                top, bottom = top_id, bottom_id
            else:
                top = pd.concat([top, top_id])
                bottom = pd.concat([bottom, bottom_id])
        if args.criterion != 'max':
            top,bottom = bottom,top   
        #breakpoint()
        #print({'k':k, 'merged': len(df), 'unq':len(unq)})

        top_context = set(top['context_id'].unique())
        bottom_context = set(bottom['context_id'].unique())
        bottom_context = bottom_context-top_context
        count = 0
        posset, negset = set([]),set([])
        for ridx, row in bottom.iterrows():
            if count >= len(unq) * k:
                break
            if row['context_id'] in bottom_context:
                if row['context_id'] in negset:
                    continue
                negset.add(row['context_id'])
                wcsample.writerow([config['dataType'][args.dataType], fileId, row['context'], row['context_id'], 'negative'])
                count+=1
        print('Negatives samples for', entry, 'are', count)
        print('Positive samples for', entry, 'are', count)
        for ridx, row in top.iterrows():
            if count<=0:
                break
            if row['context_id'] in posset:
                continue
            posset.add(row['context_id'])
            wcsample.writerow([config['dataType'][args.dataType], fileId, row['context'], row['context_id'], 'positive'])
            count-=1
        
        #breakpoint()
       
        #breakpoint()
        #break
    fsamples.close()
    return

def splitUsingAgg(args):
    print(args)
    entries = getFileEntries(args.sourceDir, args.numInputs)
    fsamples = open(os.path.join(args.outputDir, 'samples_agg_' + args.dataType +"_"+ args.field.split()[0]+ '.csv'), 'w')
    wcsample = csv.writer(fsamples)
    wcsample.writerow(['dataTypeId', 'fileId','context', 'context_id', 'label'])
    top,bottom = None, None
    for entry in entries:
        fileId = entry.split('_')[-2]
        df = pd.read_csv(os.path.join(args.sourceDir, entry))
        #breakpoint()
        unq = df.id.unique()
        f = pd.DataFrame([], columns = ['context_id', 'context']) 
        f['context_id'] = df['context_id']
        f['context']=df['context']
        f = f.drop_duplicates()
        df_agg = df.groupby('context_id').agg(agg=pd.NamedAgg(column=args.field, aggfunc=mean))
        merged = pd.merge(df_agg, f, on='context_id')
        k = args.k
        while 2*k*len(unq) > len(merged):
            k = k-1
        #print({'k':k, 'merged': len(merged), 'unq':len(unq)})
        if k!=0:
            top = merged.nlargest(k*len(unq), 'agg', keep='all')
            bottom = merged.nsmallest(k*len(unq), 'agg', keep='all')
        else:
            top = merged.nlargest(int(len(merged)/2),'agg', keep='all')
            bottom = merged.nsmallest(int(len(merged)/2),'agg', keep='all')
        if args.criterion !='max':
            top,bottom = bottom, top

        top_context = set(top['context_id'].unique())
        bottom_context = set(bottom['context_id'].unique())
        bottom_context = bottom_context-top_context
        count = 0
        posset, negset = set([]),set([])
        count = 0
        for ridx, row in bottom.iterrows():
            if k!=0 and count >= len(unq) * k:
                break
            if row['context_id'] in bottom_context:
                if row['context_id'] in negset:
                    continue
                negset.add(row['context_id'])
                wcsample.writerow([config['dataType'][args.dataType], fileId, row['context'], row['context_id'], 'negative'])
                count+=1
        print('Negatives samples for', entry, 'are', count)
        print('Positive samples for', entry, 'are', count)
        for ridx, row in top.iterrows():
            if count <= 0:
                break
            if row['context_id'] in posset:
                continue
            posset.add(row['context_id'])
            wcsample.writerow([config['dataType'][args.dataType], fileId, row['context'], row['context_id'], 'positive'])
            count-=1

    fsamples.close()
    return

def splitUsingRandom(args):
    print(args)
    entries = list(getFileEntries(args.sourceDir, args.numInputs))
    fsamples = open(os.path.join(args.outputDir, 'samples_random_' + args.dataType +"_"+ args.field.split()[0]+ '.csv'), 'w')
    wcsample = csv.writer(fsamples)
    wcsample.writerow(['dataTypeId', 'fileId','context', 'context_id', 'label'])
    print('Entries:', len(entries))
    for idx, entry in enumerate(entries):
        top,bottom = None, None
        top_context, bottom_context = None, None
        context_unq = None
        fileId = entry.split('_')[-2]
        df = pd.read_csv(os.path.join(args.sourceDir, entry))
        unq = df.id.unique()
        if len(unq) ==0:
            continue
        context_unq = set(df['context_id'].unique())
        k = args.k
        while k*len(unq) >= len(context_unq) and k>=0:
            k = k-1
        for val in unq:
            df_id = df[df['id']==val]
            if k!=0:
                top_id = df_id.nlargest(k, args.field)
                bottom_id = df_id.nsmallest(k, args.field)
            else:
                top_id = df_id.nlargest(int(df_id.shape[0]/2), args.field)
                bottom_id = df_id.nsmallest(int(df_id.shape[0]/2), args.field)
            if top is None:
                top, bottom = top_id, bottom_id
            else:
                top = pd.concat([top, top_id])
                bottom = pd.concat([bottom, bottom_id])
        if args.criterion != 'max':
            top,bottom = bottom,top   
        if k!=0:
            top_context = set(top['context_id'].unique())
        else:
            l = list(top['context_id'].unique())
            top_context = set(l[:int(len(l)/2)])
        bottom_context = context_unq-top_context
        length = min(len(top_context), len(bottom_context))
        bottom_context = np.random.choice(list(bottom_context), length, replace=False)
        count = 0
        posset, negset = set([]),set([])
        for ridx, row in df.iterrows():
            if count >= length:
                break
            if row['context_id'] in bottom_context:
                if row['context_id'] in negset:
                    continue
                negset.add(row['context_id'])
                wcsample.writerow([config['dataType'][args.dataType], fileId, row['context'], row['context_id'], 'negative'])
                count+=1
        print('Negatives samples for', entry, 'are', count)
        print('Positive samples for', entry, 'are', count)
        for ridx, row in top.iterrows():
            if count<=0:
                break
            if row['context_id'] in posset:
                continue
            posset.add(row['context_id'])
            wcsample.writerow([config['dataType'][args.dataType], fileId, row['context'], row['context_id'], 'positive'])
            count-=1
    fsamples.close()
    return


def createTrainDevTestSplit(args):
    print('Splitting into test/train/dev')
    splits = args.split.split(":") #train:dev:test
    df = pd.read_csv(os.path.join(args.outputDir, 'samples_' +args.method+'_' + args.dataType +"_"+ args.field.split()[0]+ '.csv'))
    ftrain = open(os.path.join(args.outputDir, 'train_' + args.method+'_' + args.dataType +"_"+ args.field.split()[0]+ '.csv'), 'w')
    ftest = open(os.path.join(args.outputDir, 'test_' + args.method+'_' + args.dataType +"_"+ args.field.split()[0]+ '.csv'), 'w')
    fdev = open(os.path.join(args.outputDir, 'dev_' + args.method+'_' + args.dataType +"_"+ args.field.split()[0]+ '.csv'), 'w')
    wtrain = csv.writer(ftrain)
    wtest = csv.writer(ftest)
    wdev = csv.writer(fdev)
    wtrain.writerow(['dataTypeId', 'fileId','context', 'context_id', 'label'])
    wtest.writerow(['dataTypeId', 'fileId','context', 'context_id', 'label'])
    wdev.writerow(['dataTypeId', 'fileId','context', 'context_id', 'label'])
    fileId = df.fileId.unique()
    if args.readSplit:
        print(args.splitDir)
        args.splitDir = args.splitDir if len(args.splitDir)>0 else args.outputDir
        train = np.loadtxt(os.path.join(args.splitDir, args.dataType+"_train_split.npy"))
        dev = np.loadtxt(os.path.join(args.splitDir, args.dataType+"_dev_split.npy"))
        test = np.loadtxt(os.path.join(args.splitDir, args.dataType+"_test_split.npy"))
        train = list(range(0,2000))
        dev = list(range(2000,2500))
        test = list(range(2500,3000))

    else:
        train = fileId[0: int(int(splits[0])/100*len(fileId))]
        dev = fileId[int(int(splits[0])/100*len(fileId)):int((int(splits[0])+int(splits[1]))/100*len(fileId))]
        test = fileId[int((int(splits[0])+int(splits[1]))/100*len(fileId)):]

    for idx,row in df.iterrows():
        if row['fileId'] in train:
            wtrain.writerow(row)
        elif row['fileId'] in dev:
            wdev.writerow(row)
        else:
            wtest.writerow(row)
    ftrain.close()
    fdev.close()
    ftest.close()
    print('Split done!')
    return


def getSplits(srcDir, splits, outDir, dataType):
    fileEntries = os.listdir(srcDir)
    uniqueNames = set(map(lambda x: x.split(".")[0], fileEntries))
    uniqueNames = sorted(list(uniqueNames), key=lambda x: int(x))
    train = uniqueNames[0: int(int(splits[0])/100*len(uniqueNames))]
    dev = uniqueNames[int(int(splits[0])/100*len(uniqueNames)):int((int(splits[0])+int(splits[1]))/100*len(uniqueNames))]
    test = uniqueNames[int((int(splits[0])+int(splits[1]))/100*len(uniqueNames)):]
    np.savetxt(os.path.join(outDir, dataType+"_train_split.npy"), np.asarray(train).astype(int))
    np.savetxt(os.path.join(outDir, dataType+"_test_split.npy"), np.asarray(test).astype(int))
    np.savetxt(os.path.join(outDir, dataType+"_dev_split.npy"), np.asarray(dev).astype(int))
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sourceDir', type=str, default='',
            help='Folder path to read src/tgt pairs input')
    parser.add_argument('--outputDir', type=str, default='',
            help='Folder path to write output to')
    parser.add_argument('--k', type=int, default=3,
                        help='top k')
    parser.add_argument('--dataType', type=str, default='', 
                        help='amicus/bb/minutes')
    parser.add_argument('--method', type=str, default='topk', 
                        help='Choose agg/topk for way of splitting')
    parser.add_argument('--split', type=str, default='60:20:20', 
                        help='Train:dev:test split')
    parser.add_argument('--numInputs', type=int, default='1', 
                        help='numInputs considered for context')
    parser.add_argument('--field', type=str, default='', 
                        help='field name for score')
    parser.add_argument('--criterion', type=str, default='max', 
                        help='max or min in the field')     
    parser.add_argument('--readSplit',action='store_true',help='If set, read from .npy files')  
    parser.add_argument('--splitDir', type=str, default='',
            help='Folder path to read splits from')
    parser.add_argument('--createSplit', action='store_true', help='If set, create .npy files')  
    args = parser.parse_args()
   
    if args.createSplit:
        getSplits(args.sourceDir, args.split.split(":"), args.splitDir, args.dataType)
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
    
    if args.method == 'agg':
        splitUsingAgg(args)
    elif args.method == 'topk':
        splitUsingTopK(args)
    elif args.method == 'random':
        splitUsingRandom(args)

    createTrainDevTestSplit(args)