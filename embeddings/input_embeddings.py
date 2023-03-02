import torch.nn as nn 
from gensim.models import Word2Vec
import nltk



class InputEmbedding():
    def __init__(self, sentences, batch_size= None, b_model) -> None:
        self.sentences = sentences
        self.batch_size = batch_size
        self.b_model = b_model
        
    
    def word2vec(self):
        sentences = sentences.lower()
        sentences = nltk.sent_tokenize(sentences)
        word_list = [nltk.word_tokenize(sent) for sent in sentences]
        
        wordvec = Word2Vec(word_list, min_count=1, vector_size=self.b_model)
        
        return wordvec.wv.vectors

        
        
        
        

    
