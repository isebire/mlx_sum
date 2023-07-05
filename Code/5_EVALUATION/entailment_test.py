# Evaluation dump - assuming gold and pegasus summaries

from faithfulness.Entailment import Entailment, EntailmentMethod

import pandas

entailment_metric = Entailment(method=EntailmentMethod.DOC)

source = 'Mike is a green frog who lives in a pond.'
sentence = 'Mike lives in a pond.'

entailment_result = entailment_metric.score(sentence, source)
