#python top-k.py gpt2-perplexity bb
#python top-k.py gpt2-perplexity minutes
#python top-k.py gpt2-perplexity amicus-facts
#python top-k.py gpt2-perplexity amicus

#python top-k.py roberta-entailment bb
#python top-k.py roberta-entailment minutes
#python top-k.py roberta-entailment amicus-facts
#python top-k.py roberta-entailment amicus

#python top-k.py baselines-bert bb
#python top-k.py baselines-bert minutes
#python top-k.py baselines-bert amicus-facts

#echo "Baseline bert amicus"
#python top-k.py baselines-bert amicus
#echo "Baseline bleu bb"
#python top-k.py baselines-bleu bb
#echo "Baseline bleu minutes"
#python top-k.py baselines-bleu minutes
#echo "Baseline bleu amicus facts"
#python top-k.py baselines-bleu amicus-facts
echo "Baseline bleu amicus"
#
python top-k.py baselines-bleu amicus
