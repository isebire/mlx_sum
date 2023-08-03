import jsonlines
import logging
import numpy as np
from tqdm import tqdm
import os
import re
import csv
#import matplotlib.pyplot as plt
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)
gpu = True

def preprocessSentence(str_):
    str_ = str_.replace("<t>", "")
    str_ = str_.replace("</t>", "")
    regex = re.compile('[*▪_\"\"•]') 
    str_ = re.sub(regex,'', str_)
    str_ = str_.replace('\n', '').replace('\r', '')
    return str_

def getExamples(path):
    f = jsonlines.open(path, "r")
    pairs = []
    while True:
        try:
            sample = f.read()
            if sample['density_bin'] == 'abstractive':
                s,t = sample['text'], sample['summary']
                pairs.append((s,t))
        except EOFError:
            break
    logger.info(f'Got all examples!')
    return np.array(pairs[:5000])

if __name__=="__main__":
    path = "data/newsroom/news.jsonl"
    out_root = "data/newsroom"
    splitDir = "classifier-dumps"
    pairs = getExamples(path)
    np.random.seed(100)
    idx = np.random.choice(np.arange(len(pairs)), 3000, replace=False)
    pairs = pairs[idx]
    source =  open(os.path.join(out_root, "newsroom.src"), "w")
    target = open(os.path.join(out_root, "newsroom.tgt"), "w")
    for idx, p in enumerate(pairs):
        source.write(preprocessSentence(p[0])+"\n")
        target.write(preprocessSentence(p[1])+"\n")
    source.close()
    target.close()
    np.save(os.path.join(splitDir, 'newsroom_train_split.npy'), np.arange(0,2000))
    np.save(os.path.join(splitDir,'newsroom_dev_split.npy'), np.arange(2000,2500))
    np.save(os.path.join(splitDir,'newsroom_test_split.npy'), np.arange(2500,3000))
    


    



    
    