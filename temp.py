# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from instrument_utils import *
import pandas as pd
import pickle

export_path = 'C:/Users/徐泽玮/Desktop/Patrick program/week7/dataset/'

###STEP ONE cleaning
review_data = open_csv()
new_review = clean_txt(review_data['reviewText'][0:])#对数据进行清洗

#the_tf_idf = my_tf_idf(new_review.review) #tf-idf Transformation
#new_review['review_stem'] = new_review.review.apply(my_stem_fun)#寻找单词的词根

dump_object(new_review,"all_data", export_path) #first time run    
new_review = pickle.load(open(export_path + "all_data" + ".pkl", "rb"))

review_new = new_review.drop(0)
dump_object(review_new,"review_new", export_path)#存入到new_matrix里面

###STEP TWO calculating
matrix1 = text_score(new_review['review'],"weighted",export_path)#利用sentiwrodnet计算所有词对应的senti_weight
matrix1 = pickle.load(open(export_path + "weighted" + ".pkl", "rb"))#下载
dump_object(matrix1,"senti_weight_1", export_path)#存入到new_matrix里面
#将DataFrame存储为csv,index表示是否显示行名，default=True
matrix1.to_csv("C:/Users/徐泽玮/Desktop/Patrick program/week7/dataset/test.csv",index=False,sep=',')#写入csv

#new_matrix = matrix1[matrix1.part.isin(['n', 'v','a','r'])]#把part为空的去掉
#dump_object(new_matrix,"new_matrix", export_path)#存入到new_matrix里面
#new_matrix = pickle.load(open(export_path + "new_matrix" + ".pkl", "rb"))

###STEP THREE senti_weight
senti_dict = get_senti_weight(matrix1,matrix1['key'])#senti_weight这一列的命名必须为score
dump_object(senti_dict,"single_senti_weight_1", export_path)#保存为pickle文件
senti_dict = pickle.load(open(export_path + "single_senti_weight_1" + ".pkl", "rb"))
senti_dict = pickle.load(open(export_path + "single_senti_weight_1" + ".pkl", "rb"))
print(senti_dict[senti_dict['key'] == 'abnormalities'].average)



review_new['score'] = None
for i in range(len(new_review['overall'])):
    if float(new_review['overall'][i]) >= 3:
        review_new['score'][i] = 1
    else:
        review_new['score'][i] = 0

positive =  sum(review_new['score'] == 1)
negative =  sum(review_new['score'] == 0)










