import spacy
import argparse
import os
import numpy as np
nlp = spacy.load('en_core_web_sm')
"""
Installing spacy: pip install spacy
Installing en_core_web_sm : python -m spacy download en_core_web_sm
"""
def retreiveSentences(document):
    about_doc = nlp(document)
    sentences = list(about_doc.sents)
    return sentences

def retreiveSrcTgtPairs(srcFilePath, tgtFilePath):
    print('Retrieving Source,Target pairs')
    fsource = open(srcFilePath,"r")
    ftarget = open(tgtFilePath, "r")
    source = fsource.read().split("\n")
    target = ftarget.read().split("\n")
    assert len(source)==len(target)
    fsource.close()
    ftarget.close()
    return list(zip(source,target))

def processBeige(args):
    pairs = retreiveSrcTgtPairs(args.source, args.target)
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
    for idx, entry in enumerate(pairs):
        print(idx)
        sourceSent = np.array(retreiveSentences(entry[0]))
        targetSent = np.array(retreiveSentences(entry[1]))
        with open(os.path.join(args.outputDir, str(idx)+".src"), "w") as outfile:
            outfile.write('\n'.join(str(v) for v in sourceSent))
            outfile.close()
        with open(os.path.join(args.outputDir, str(idx)+".tgt"), "w") as outfile:
            outfile.write("\n".join(str(v) for v in targetSent))
            outfile.close()
    print('Done creating sentence file dumps!')

def processAmicus(args):
    fileEntries = os.listdir(args.sourceDir)
    uniqueNames = set(map(lambda x: x.split(".")[0], fileEntries))
    uniqueNames = sorted(list(uniqueNames))
    for name in uniqueNames:
        print(name)
        if name == 'missing':  # if filenames don't start with 0.src
            continue
        fsource = open(os.path.join(args.sourceDir, name+".src"),"r")
        ftarget = open(os.path.join(args.sourceDir, name+".tgt"), "r")
        source = fsource.read()
        target = ftarget.read()
        sourceSent = np.array(retreiveSentences(source))
        targetSent = np.array(retreiveSentences(target))
        fsource.close()
        ftarget.close()
        with open(os.path.join(args.outputDir, name+".src"), "w") as outfile:
            outfile.write('\n'.join(str(v) for v in sourceSent))
            outfile.close()
        with open(os.path.join(args.outputDir, name+".tgt"), "w") as outfile:
            outfile.write("\n".join(str(v) for v in targetSent))
            outfile.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='',
            help='File path to .src file, for example: gs-beige.src')
    parser.add_argument('--target', type=str, default='',
            help='File path to .tgt file, for example: gs-beige.tgt')
    parser.add_argument('--sourceDir', type=str, default='',
            help='Folder path to read src and tgt files for amicus type files')
    parser.add_argument('--outputDir', type=str, default='',
            help='Folder path to write output to')
    parser.add_argument('--datatype', type=str, default='beige',
            help='Folder path to write output to')
    args = parser.parse_args()
    
    if args.datatype in ['beige', 'newsroom']:
        processBeige(args)
    elif args.datatype in ['amicus', 'pubmed', 'arxiv']:
        processAmicus(args)
    else:
        print('Incorrect type!')
