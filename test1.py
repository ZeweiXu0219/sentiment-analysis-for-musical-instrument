# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:08:36 2021

@author: 徐泽玮
"""
import pickle
import numpy as np
import collections
import torch
import pandas as pd
import enchant
import csv
import pandas as pd
import nltk
from nltk import word_tokenize #分词函数
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn #得到单词情感得分
import re
#import enchant
import string #本文用它导入标点符号，如!"#$%& 
global sw
global d
sw = stopwords.words('english')
d = enchant.Dict("en_US")
import gensim.models.word2vec as word2vec


review = pickle.load(open('C:/Users/徐泽玮/Desktop/Patrick program/week9/data/all_data.pkl', "rb"))
matrix1 = pickle.load(open('C:/Users/徐泽玮/Desktop/Patrick program/week9/data/weighted.pkl', "rb"))
senti_dict = pickle.load(open('C:/Users/徐泽玮/Desktop/Patrick program/week9/data/single_senti_weight_1.pkl', "rb"))

#test
#print(review.head())
#print(matrix1.head())
#print(senti_dict.head())

#word2vec

#sent = word2vec.Text8Corpus('/content/text8')
#tmp = word2vec.Word2Vec(sent,size=200)
#tmp.save('/content/text.model')
model = word2vec.Word2Vec.load('C:/Users/徐泽玮/Desktop/Patrick program/week9/data/text2020.model')
#test
#print(model['soldered']) 
#word_list = [word for word in senti_dict['key'] if word in mylist]

 
def process_word_vectors():
    import numpy as np
    vecs = []
    for word in senti_dict['key']:
        try:
            vec = model[word]
            vecs.append(vec)
        except:
            vec = np.ones(200)
            vecs.append(vec)
            pass 
    vecs = np.array(vecs)
    np.savetxt("C:/Users/徐泽玮/Desktop/Patrick program/week9/data/vecs.txt", vecs)

def get_word_vectors():
    import numpy as np
    vecs = np.loadtxt("C:/Users/徐泽玮/Desktop/Patrick program/week9/data/vecs.txt")
    word2vec = {}
    word_list = senti_dict['key']
    for i in range(len(word_list)):
        word2vec[word_list[i]] = vecs[i]
    return word2vec


def get_sentiment_dict():
    sentiment_dict = {}
    for word in senti_dict.key:
        sentiment_dict[word] = float(senti_dict[senti_dict['key'] == word].average)#建立dict
    return sentiment_dict


def get_weighted_word_vectors():
    word2vec = get_word_vectors()
    sentiment_dict = get_sentiment_dict()
    word2Vec = {}
    for i in word2vec.keys():
        word2vec[i] = sentiment_dict[i] * word2vec[i]
    return word2vec
    #senti_dict[senti_dict['key'] == i].average


def get_data():                      #clean txt
  new_review = pd.DataFrame(None,columns=['review'])
  for word in review['reviewText'][0:]:    #file_in = review_data['reviewText'][0:]
    tmp = re.sub("[^A-z']+", ' ', word).lower()
    tmp = [word for word in tmp.split() if word not in sw] #tokenization
    tmp = [word for word in tmp if d.check(word)] #filter non-English words
    new_review = new_review.append({'review':tmp},ignore_index = True)
  return new_review['review'].to_list()

def process_data(sentence_length, words_size, embed_size):
    #normalize_sentiment_words()
    sentences = get_data()
    #sentences = review['review'].to_list()
    frequency = collections.Counter()
    for sentence in sentences:
        for word in sentence:
            frequency[word] += 1
    word2index = dict()
    for i, x in enumerate(frequency.most_common(words_size)):
        word2index[x[0]] = i + 1
    word2vec = get_weighted_word_vectors()
    word_vectors = torch.zeros(words_size + 1, embed_size)
    for k, v in word2index.items():
        try:
          word_vectors[v,:] = torch.from_numpy(word2vec[k])
        except:
          #word_vectors[v[0:1],:] = torch.from_numpy(word2vec[k])
          pass

    rs_sentences = []
    for sentence in sentences:
        sen = []
        for word in sentence:
            if word in word2index.keys():
                sen.append(word2index[word])
            else:
                sen.append(0)
        if len(sen) < sentence_length:
            sen.extend([0 for _ in range(sentence_length - len(sen))])
        else:
            sen = sen[:sentence_length]
        rs_sentences.append(sen)
    label = [1 for _ in range(9793)]                   #positive一共9793条review
    label.extend([0 for _ in range(467)])                 #一共467条review
    label = np.array(label)
    print(label)
    return rs_sentences, label, word_vectors


def make_list(matrix_name,column_name,epochs):
    import pandas as pd
    matrix_name = pd.DataFrame()
    for i in range(epochs):
        matrix_name[i] = [number for number in matrix_name[column_name][411*i:(411*(i+1))-1]]
    return matrix_name

def dump_object(obj_in, name_in, path_out):
    import pickle
    pickle.dump(obj_in, open(path_out + name_in + ".pkl", "wb"))
    return 0

