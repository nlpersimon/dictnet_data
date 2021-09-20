from functools import partial
import jsonlines
from gensim.models import KeyedVectors
import os

dirname = os.path.dirname(__file__)
join_path = partial(os.path.join, dirname)


with jsonlines.open(join_path('data/cambridge.sense.000.jsonl')) as f:
    senses = list(f)

def_embeds = KeyedVectors.load_word2vec_format(
    join_path('data/cambridge.sense.000.sg.def_embeds.txt'), binary=False)
