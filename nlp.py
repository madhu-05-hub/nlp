pip install nltk

import nltk
from nltk import FreqDist, bigrams, trigrams
from nltk.util import ngrams
from nltk.corpus import reuters
from collections import defaultdict, Counter
nltk.download('reuters')
nltk.download('punkt')

def print_first_100_chars():
    file_ids = reuters.fileids()[:5]
    for file_id in file_ids:
        print(f"\n--- {file_id} ---")
        document = reuters.raw(file_id)
        print(document[:100])  # Print the first 100 characters of each document

print_first_100_chars()

def get_unigrams(corpus):
    tokenized_text = [word.lower() for word in word_tokenize(corpus) if word.isalpha()]
    unigrams = FreqDist(tokenized_text)
    return unigrams

corpus = ' '.join(reuters.raw(fileid) for fileid in reuters.fileids()[:10])  # Sample corpus
unigrams = get_unigrams(corpus)

print("Unigrams:")
for unigram, freq in unigrams.items():
    print(f"{unigram}: {freq}")

def get_bigrams(corpus):
    tokenized_text = [word.lower() for word in word_tokenize(corpus) if word.isalpha()]
    bigram_list = list(bigrams(tokenized_text))
    bigrams_freq = FreqDist(bigram_list)
    return bigrams_freq

# Sample corpus
corpus = ' '.join(reuters.raw(fileid) for fileid in reuters.fileids()[:5])
bigrams_freq = get_bigrams(corpus)

print("Bigrams:")
for bigram, freq in bigrams_freq.items():
    print(f"{bigram}: {freq}")

def get_trigrams(corpus):
    tokenized_text = [word.lower() for word in word_tokenize(corpus) if word.isalpha()]
    trigram_list = list(trigrams(tokenized_text))
    trigrams_freq = FreqDist(trigram_list)
    return trigrams_freq

# Sample corpus
corpus = ' '.join(reuters.raw(fileid) for fileid in reuters.fileids()[:5])
trigrams_freq = get_trigrams(corpus)

print("Trigrams:")
for trigram, freq in trigrams_freq.items():
    print(f"{trigram}: {freq}")

def get_bigram_probabilities(corpus):
    tokenized_text = [word.lower() for word in word_tokenize(corpus) if word.isalpha()]
    bigram_list = list(bigrams(tokenized_text))
    bigrams_freq = FreqDist(bigram_list)
    unigrams_freq = FreqDist(tokenized_text)
    
    bigram_prob = {}
    for (w1, w2), freq in bigrams_freq.items():
        bigram_prob[(w1, w2)] = freq / unigrams_freq[w1]
    return bigram_prob

# Sample corpus
corpus = ' '.join(reuters.raw(fileid) for fileid in reuters.fileids()[:5])
bigram_prob = get_bigram_probabilities(corpus)

print("Bigram Probabilities:")
for bigram, prob in bigram_prob.items():
    print(f"{bigram}: {prob:.4f}")


def next_word_prediction(bigram_prob, word):
    candidates = {w2: prob for (w1, w2), prob in bigram_prob.items() if w1 == word}
    if not candidates:
        return "No prediction available"
    return max(candidates, key=candidates.get)

corpus = ' '.join(reuters.raw(fileid) for fileid in reuters.fileids()[:5])
bigram_prob = get_bigram_probabilities(corpus)

sentence = "The stock market"
last_word = sentence.split()[-1]

prediction = next_word_prediction(bigram_prob, last_word)

print(f"Sentence: '{sentence}'")
print(f"Next word prediction for '{last_word}': {prediction}")
