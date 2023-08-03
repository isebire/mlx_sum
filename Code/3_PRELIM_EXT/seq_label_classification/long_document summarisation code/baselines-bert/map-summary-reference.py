import argparse
import os, sys
sys.path.insert(0, os.path.abspath('..'))
os.environ["PYTHONIOENCODING"] = "utf-8"
from utils.docutils import processData


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true',help='If set, use the GPU')
    parser.add_argument('--sourceDir', type=str, default='',
            help='Folder path to read src/tgt pairs input')
    parser.add_argument('--outputDir', type=str, default='',
            help='Folder path to write output to')
    parser.add_argument('--length', type=int, default=0,
                        help='Pairs of documents to consider')
    parser.add_argument('--numInputs', type=int, default=1,
                        help='Number of inputs sentences to use as context')
    args = parser.parse_args()

    processData(args, model_type="baselines-bert")