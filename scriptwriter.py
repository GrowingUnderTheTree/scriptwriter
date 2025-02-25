import nltk
import random
from nltk.util import ngrams
from collections import defaultdict, Counter
nltk.download('punkt_tab')

f=open('scripts.csv',encoding="utf-8")
raw=f.read()

tokens = nltk.word_tokenize(raw)
trigrams = list(ngrams(tokens, 3))

trigram_model = defaultdict(list)
for w1, w2, w3 in trigrams:
    trigram_model[(w1, w2)].append(w3)

trigram_probabilities = {key: Counter(value) for key, value in trigram_model.items()}

def generate_text(seed, num_words=5000):
    w1, w2 = seed.split()
    generated_text = [w1, w2]

    for _ in range(num_words):
        next_word_options = trigram_probabilities.get((w1, w2), None)
        if not next_word_options:
            break  # Stop if no predictions available
        w3 = random.choices(list(next_word_options.keys()), weights=next_word_options.values())[0]
        generated_text.append(w3)
        w1, w2 = w2, w3  # Move to next trigram context

    return ' '.join(generated_text)