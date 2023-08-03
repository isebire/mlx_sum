import logging
import pytextrank
import spacy
import sys
import os
from tqdm import tqdm

"""
Code written following instructions https://github.com/DerwenAI/pytextrank/blob/master/example.py
and https://pypi.org/project/pytextrank/

"""

nlp = spacy.load("en_core_web_sm")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class TextRankSummariser:
    def __init__(self, srcPath, tgtPath, outPath, compare=False, top=70) -> None:
        """
        srcPath: Path to file containing the sources as line seperated examples
        tgtPath: Path to file containing the targets as line seperated examples
        outPath: Path to where the output summaries should be written as line seperated examples
        compare: If set to true, the sizes of summaries should be comparable to the mentioned array
        top: Argument that can be used to determine the summary length
        """
        self.srcPath = srcPath 
        self.tgtPath = tgtPath
        self.outPath = outPath
        self.source = []
        self.target = []
        self.predicted = []
        self.top = top 
        self.compare = compare
        logger.info('Class intialised.')
        self.compare_sizes = []
        self.load_data()
    
    
    def load_data(self) -> None:
        f = open(self.srcPath, "r")
        self.source = f.read().split("\n")
        f.close()
        f = open(self.tgtPath, "r")
        self.target = f.read().split("\n")
        f.close()
        logger.info(f'Loaded data from {self.srcPath}, {self.tgtPath}.')

    def setupPipeline(self) -> None:
        # Need to add PyTextRank into the spaCy pipeline
        tr = pytextrank.TextRank(logger=logger)
        nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
        logger.info(f'Setup the pipeline {nlp.pipe_names}')
    
    def getControlSizes(self, filepath) -> None:

        def retreiveSentences(document):
            about_doc = nlp(document)
            sentences = list(about_doc.sents)
            return len(sentences)

        f = open(filepath, "r")
        sent = f.read().split("\n")
        f.close()
        
        self.compare_sizes = list(map(retreiveSentences, sent))
        logger.info(f'Retreived the comparable sizes from {filepath}')


    
    def __call__(self):
        # Callable method that does the summarization
        logger.info('Summarizer called!')
        for i, s in tqdm(enumerate(self.source)):
            doc = nlp(s) # parse the document
            txt = ""
            t = self.top if not self.compare else self.compare_sizes[i]
            for sent in doc._.textrank.summary(limit_phrases=25, limit_sentences=self.top):
                txt+=str(sent)
            self.predicted.append(txt)

        logger.info('Summaries generated!')
        f = open(self.outPath, "w")
        f.write("\n".join(self.predicted))
        f.close()
        logger.info(f'Output written to {self.outPath}')

if __name__ == "__main__":

    #TODO: Code can be modified to use an argparser. 
    src = "test/test-amicus-combined.src"
    tgt = "test/test-amicus-combined.tgt"
    out = "test/test-amicus-textrank.hypo"
    agg = "test/amicus-preplexity-agg.src"
    summ = TextRankSummariser(src, tgt, out, compare=True)
    summ.setupPipeline()
    summ.getControlSizes(agg)
    summ()




                
            


    

    


