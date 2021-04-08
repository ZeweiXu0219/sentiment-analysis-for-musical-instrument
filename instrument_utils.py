# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:12:16 2021

@author: 徐泽玮
"""
import csv
import pandas as pd
import nltk
from nltk import word_tokenize #分词函数
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn #得到单词情感得分
import re
import enchant
import string #本文用它导入标点符号，如!"#$%& 
global sw
global d
sw = stopwords.words('english')
d = enchant.Dict("en_US")


def open_csv():
    tmp_lst = []
    with open('C:/Users/徐泽玮/Desktop/Patrick program/week7/dataset/archive/Musical_instruments_reviews.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            tmp_lst.append(row)
    df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
    review_data = pd.DataFrame({'reviewText':df['reviewText'],'review':None,'overall':df['overall']})
    return review_data

def my_tf_idf(df_in):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    my_vec = pd.DataFrame(vectorizer.fit_transform(df_in).toarray())
    my_vec.columns = vectorizer.get_feature_names()
    return my_vec

def clean_txt(file_in):
    new_review = pd.DataFrame(None,columns=['review'])
    for word in file_in:    #file_in = review_data['reviewText'][0:]
        tmp = re.sub("[^A-z']+", ' ', word).lower()
        tmp = [word for word in tmp.split() if word not in sw] #tokenization
        tmp = [word for word in tmp if d.check(word)] #filter non-English words
        tmp = ' '.join(tmp)
        new_review = new_review.append({'review':tmp}, ignore_index = True)
    tmp_data = open_csv()
    tmp_data['review'] = [sentence for sentence in new_review['review']] #添加一列到dataframe中
    return tmp_data

def my_stem_fun(str_in):
    from nltk.stem.porter import PorterStemmer
    my_stem = PorterStemmer()
    the_out = [my_stem.stem(word) for word in str_in.split()]
    the_out = ' '.join(the_out)
    return the_out

def dump_object(obj_in, name_in, path_out):
    import pickle
    pickle.dump(obj_in, open(path_out + name_in + ".pkl", "wb"))
    return 0

def text_score(file_in,name_in,path_out):
    #create单词表
    #nltk.pos_tag是打标签
    #matrix = pd.DataFrame()
    #for sentence in file_in:
        #text = sentence
    list1 = file_in.tolist()
    text = ' '.join(list1)
    ttt = nltk.pos_tag([i for i in word_tokenize(str(text).lower()) if i not in sw])
    word_tag_fq = nltk.FreqDist(ttt)
    wordlist = word_tag_fq.most_common()
    
        #变为dataframe形式
    key = []
    part = []
    frequency = []
    for i in range(len(wordlist)):
        key.append(wordlist[i][0][0])
        part.append(wordlist[i][0][1])
        frequency.append(wordlist[i][1])
    textdf = pd.DataFrame({'key':key,
                      'part':part,
                      'frequency':frequency},
                      columns=['key','part','frequency'])

    #编码
    n = ['NN','NNP','NNPS','NNS','UH']
    v = ['VB','VBD','VBG','VBN','VBP','VBZ']
    a = ['JJ','JJR','JJS']
    r = ['RB','RBR','RBS','RP','WRB']

    for i in range(len(textdf['key'])):
        z = textdf.iloc[i,1]

        if z in n:
            textdf.iloc[i,1]='n'
        elif z in v:
            textdf.iloc[i,1]='v'
        elif z in a:
            textdf.iloc[i,1]='a'
        elif z in r:
            textdf.iloc[i,1]='r'
        else:
            textdf.iloc[i,1]=''
            
        #计算单个评论的单词分数
    score = []
    for i in range(len(textdf['key'])):
        m = list(swn.senti_synsets(textdf.iloc[i,0],textdf.iloc[i,1]))
        s = 0
        ra = 0
        if len(m) > 0:
            for j in range(len(m)):
                s += (m[j].pos_score()-m[j].neg_score())/(j+1)
                ra += 1/(j+1)                                        #计算权重
            score.append(s/ra)
        else:
            score.append(0)
    #matrix_origin = pd.concat([textdf,pd.DataFrame({'score':score})],axis=1)
    #matrix = matrix.append(matrix_origin,ignore_index=True)  
    matrix = pd.concat([textdf,pd.DataFrame({'score':score})],axis=1)
    a = sum(matrix['frequency'])
    matrix['frequency'] = matrix['frequency']/a
    dump_object(matrix, name_in, path_out)           
    return matrix#pd.concat([textdf,pd.DataFrame({'score':score})],axis=1)

def get_senti_weight(file_in,file_in_key):#file_in是输入矩阵的名字#file_in_key是词典列#输入的矩阵的senti_weight的列名必须为score
    senti_dict = pd.DataFrame(None,columns = ['key'])
    a = list(set(file_in_key))#提取不重复的词
    for word in a:
        senti_dict = senti_dict.append({'key':word},ignore_index=True)#写入senti_dict
        
    senti_dict2 = pd.DataFrame(None,columns = ['average'])
    for word in senti_dict['key']:
        average_ = sum(file_in[file_in_key == word].score)/len(file_in[file_in_key == word].score)#计算每个词，所有词性senti_weight的平均值
        senti_dict2 = senti_dict2.append({'average':average_},ignore_index=True)
    senti_dict['average'] = [num for num in senti_dict2['average']]#将senti_weight与文字对应
    for i in range(len(senti_dict)):
        if senti_dict['average'][i] == 0:#将所有senti_weight = 0的senti_weight替换成1
            senti_dict['average'][i] =1
    return senti_dict






