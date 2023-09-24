from language_model import *
from nltk.tokenize import word_tokenize

review1 = 'I had a great time, their apples were to die for.'
review2 = 'It was a horrible experience, Ill never go again.'
review3 = 'It was phenomenally mid.'
reviews = [review1, review2, review3]

bos_token, eos_token = '<s>', '</s>'
ngram_size = 3 # Use trigrams
word_lm = NGramLM(bos_token, eos_token, word_tokenize, ngram_size)
word_lm.get_ngram_counts(reviews)

def unigram_tests(word_lm: NGramLM):
    # print(word_lm.ngram_count[0]))
    
    print(f"Number of unigrams: {len(word_lm.ngram_count[0])}")
    least_unigram = min(word_lm.ngram_count[0].keys(), key=lambda x: word_lm.ngram_count[0][x])
    print(f"Unigram with smallest count: {least_unigram}\tCount: {word_lm.ngram_count[0][least_unigram]}")
    print(f"Unknown unigram: {word_lm.ngram_count[0]['UNK']}")

unigram_tests(word_lm)

print(f"Number of BOS token: {word_lm.ngram_count[0][bos_token]}")
print(f"Number of bigrams: {len(word_lm.ngram_count[1])}")

