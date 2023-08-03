import os
import numpy as np
import argparse
import csv
import re
import pdb

def processtxt(source):
    source = source.replace('\n', '').replace('\r', '')
    print('Removing table of contents')
    start = ['INTEREST OF AMICUS AMERICAN BAR ASSOCIATION','INTEREST OF AMICUS CURIAE', 'The American Bar Association \\("ABA"\\)','THE AMERICAN BAR ASSOCIATION','STATEMENT OF INTEREST','INTEREST OF THE AMICUS CURIAE']
    end = ['Counsel for Amicus Curiae','Counsel for Amicus Curiae','Reprinted with permission', 'Counsel of Record', 'APPENDIX','CERTIFICATE OF SERVICE','ASSOCIATION AS AMICUS CURIAE']
    #print(start,end)
    result = None
    for s in start:
        for e in end:
            searchkey = s+"(.*)"+e
            r = re.search(searchkey, source)
            if r:
                result = r
                text = s
                break
    article, summary = None, None
    if result is not None:
        source = result.group(1)
        source = source.split(text)[-1]
        print('Extracting summary')
        start = ['SUMMARY OF ARGUMENT', 'SUMMARY OF THE ARGUMENT']
        end = ['ARGUMENT','STATES HAVE A DUTY TO PROTECT JUDICIAL']
        result = None
        sttext =None
        for s in start:
            for e in end:
                searchkey = s+"(.*)[\.|\s|\-|\d]"+e
                r = re.search(searchkey, source)
                if r:
                    result = r
                    sttext = s
                    text = e
                    break
        summary = result.group(1).split(sttext)[-1]
        print('Extracting article')
        result = re.search(r'[\.|\s|\-|\d]'+text+'(.*)', source)
        article = result.group(1).split(text)[-1]

    print(len(article), len(summary))
    return article, summary
    


def getFilesContent(dir_, outdir_):
    print('GetFileContent called')
    entries = sorted(os.listdir(dir_))
    outentries = sorted(os.listdir(outdir_))
    fmap = open(os.path.join(outdir_, 'mapping.csv'), 'w')
    writer = csv.writer(fmap)
    writer.writerow(['Filename', 'idx'])
    content = []
    missing = []
    print(outentries)
    for idx, entry in enumerate(entries):
        writer.writerow([entry, idx])
        if str(idx)+".src" in outentries: #
            continue
        writer.writerow([entry, idx])
        fsource = open(os.path.join(dir_, entry),"r",encoding="latin-1")
        source = fsource.read()
        try:
            print('Processing', entry, 'with idx', idx)
            article, summary = processtxt(source)
            if article and summary:
                fsrc = open(os.path.join(outdir_, str(idx) + ".src"),"a+")
                ftgt = open(os.path.join(outdir_, str(idx) + ".tgt"),"a+")
                fsrc.write(article)
                ftgt.write(summary)
                fsrc.close()
                ftgt.close()
            else:
                print('Failed to process', entry, 'with idx', idx)
                missing.append(entry)
        except:
            print('Failed to process', entry, 'with idx', idx)
            missing.append(entry)
        fsource.close()
    print('Missing...', len(missing))
    fmap.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourceDir', type=str, default='',
            help='Folder path to read amicus txt files')
    parser.add_argument('--outputDir', type=str, default='./amicus_processed',
            help='Folder path to write amicus txt files')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)


    getFilesContent(args.sourceDir, args.outputDir)
