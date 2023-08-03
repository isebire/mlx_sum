import numpy as np
import sys
import tqdm

from utils import Bcolors

with open("source.txt", "r") as f:
    source = f.read().strip().split("\n")

with open("bart.txt", "r") as f:
    bart = f.read().strip().split("\n")

assert len(source) == len(bart)

for ngram in [1, 2, 3, 4, 5, 6]:
    output_overlap = ""

    for s1, b1 in zip(source, bart):
        print_pointer = 0

        output_overlap += Bcolors.BOLD + "SOURCE = " + Bcolors.ENDC + s1 + "\n\n"

        tokens = b1.split()
        ngram_list = [" ".join(tokens[i:i + ngram]) for i in range(0, len(tokens) - ngram + 1, 1)]

        s1 = " " + " ".join(s1.split()) + " "
        b1 = " " + " ".join(b1.split()) + " "

        for ngram_starter, word_seq in enumerate(ngram_list):
            s1_presence = word_seq in s1
            b1_presence = word_seq in b1
	
            num_print_tokens = ngram_starter + ngram - print_pointer
            print_tokens = " ".join(word_seq.split()[-num_print_tokens:])

            if s1_presence and b1_presence:
                output_overlap += Bcolors.OKGREEN + print_tokens + Bcolors.ENDC + " "
                print_pointer += num_print_tokens
            elif not s1_presence and b1_presence:
                output_overlap += Bcolors.ENDC + print_tokens + Bcolors.ENDC + " "
                print_pointer += num_print_tokens

            if ngram_starter == print_pointer:
                output_overlap += tokens[print_pointer] + " "
                print_pointer += 1

        if print_pointer < len(tokens):
            output_overlap += tokens[print_pointer] + " "
            print_pointer += 1

        output_overlap += "\n\n"

    with open("overlap_%dgram.txt" % ngram, "w") as f:
        f.write(output_overlap)
