# EECS 487 Intro to NLP
# Assignment 1

import math
import random
from collections import defaultdict
from itertools import product
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

should_remove_half_BOS_tokens = True

class NGramLM:
    """N-gram language model."""

    def __init__(self, bos_token, eos_token, tokenizer, ngram_size):
        self.ngram_count = {}
        for i in range(ngram_size):
            # Each of these will be its own dictionary containing all of the (i-1)-grams
            # For instance, self.ngram_count[0] will be its own dictionary containing all unigrams
            self.ngram_count[i] = defaultdict(int)

        self.ngram_size = ngram_size

        self.vocab_sum = None # could be useful in linear interpolation
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.tokenizer = tokenizer

    def get_ngram_counts(self, reviews):

        # Adding BOS and EOS tokens to raw text  
        raw_text = ''
        bos_tokens = [ self.bos_token for _ in range( self.ngram_size - 1)] if self.ngram_size > 1 else None
        tokens = []    
        
        for review in reviews:
            tokens += bos_tokens
            tokens += self.tokenizer(review)  
            tokens += [ self.eos_token ]
        
        # Count frequencies 
        token_indexes = defaultdict(list)        
        for i, token in enumerate(tokens):
            token_indexes[token].append(i) 
            
        # Modify for unknown
        for token, indexes in token_indexes.items():
            if len(indexes) >= 2:
                continue
        
            for index in indexes:
                tokens[index] = 'UNK'
            
        # Build unigram model 
        for token in tokens:
            self.ngram_count[0][token] += 1
            
        # Build n-gram models
        for i in range(1, self.ngram_size):
            self.build_ngram_counts(i, tokens)        
        
        
    def build_ngram_counts(self, size: int, tokens: list) -> None:
        n = len(tokens)
        for i in range(n-size):
            ngram = ( tokens[i+j] for j in range(size) ) if size > 1 else tokens[i]    
            self.ngram_count[size][ngram] +=1
            
            
    def add_k_smooth_prob(self, n_minus1_gram, unigram, k):
        probability = 0

        #################################################################################
        # TODO: calculate probability using add-k smoothing
        #################################################################################

        #################################################################################

        return probability

    def linear_interp_prob(self, n_minus1_gram, unigram, lambdas):
        probability = 0

        #################################################################################
        # TODO: calculate probability using linear interpolation
        #################################################################################

        #################################################################################

        return probability
    
    def get_probability(self, n_minus1_gram, unigram, smoothing_args):
        probability = 0

        #################################################################################
        # TODO: calculate probability using add-k smoothing or linear interpolation
        #################################################################################

        #################################################################################

        return probability
    
    def get_perplexity(self, text, smoothing_args):
        perplexity = 0

        #################################################################################
        # TODO: calculate perplexity for text
        #################################################################################

        #################################################################################

        return perplexity
    
    def search_k(self, dev_data):
        best_k = 0

        #################################################################################
        # TODO: find best k value
        #################################################################################

        #################################################################################

        return best_k
    
    def search_lambda(self, dev_data):
        best_lambda = [0, 0, 0]

        #################################################################################
        # TODO: find best lambda values
        #################################################################################

        #################################################################################

        return best_lambda
    
    def generate_text(self, prompt, smoothing_args):
        generated_text = prompt.copy()

        #################################################################################
        # TODO: generate text based on prompt
        #################################################################################

        #################################################################################

        print(' '.join(generated_text))

def load_new_data(df):

    df_class1 = None
    df_class2 = None

    #################################################################################
    # TODO: load the reviews based on a split of your choosing
    #################################################################################
    
    split_name = ""

    #################################################################################

    display(df_class1[["text", split_name]])
    display(df_class2[["text", split_name]])

    return (df_class1.text, df_class2.text)
    

def predict_class(test_file, class1_lm, class2_lm, smoothing_args):

    class1_ppl = 0
    class2_ppl = 0

    #################################################################################
    # TODO: load the review in test_file, predict its class
    #################################################################################

    #################################################################################

    print(f"Perplexity for class1_lm: {class1_ppl}")
    print(f"Perplexity for class2_lm: {class2_ppl}")
    if class1_ppl < class2_ppl:
        print("It is in class 1")
    else:
        print("It is in class 2")
