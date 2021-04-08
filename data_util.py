import jieba
import numpy as np
import collections
import torch


def process_sentiment_words():
    f = open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/sentiment_words.txt", 'w', encoding='utf-8')
    with open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/sentiment_words.csv", 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines[1:]:
            line = line.strip().replace(' ', '').split(',') #删掉行首尾换行符.把空格替换成无.将其按，分开
            if line[1] == 'idiom':
                continue
            if line[6] == '1.0':#当第七列为1时
                f.write(line[0] + ',' + str(line[5]) + '\n')#将第一列和第六列的内容写入txt文件
            elif line[6] == '2.0':#当第七列为2时
                f.write(line[0] + ',' + str(-1 * float(line[5])) + '\n')#将第一列和第六列的相反数写入txt文件
    f.close()#将其文本关闭


def normalize_sentiment_words():
    #process_sentiment_words()
    words = []
    weights = []
    with open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/sentiment_words.txt", 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split(',')#删去换行符，并且按照，分开
            words.append(line[0])#将第一列加入到words这个list里面
            weights.append(float(line[1]))#将第二列变成float然后添加到weights这个list里面
    weights = np.array(weights)#把weights变成array
    mean = weights.mean()#取平均数
    std = weights.std()#计算标准差
    weights = (weights - mean)/std#标准化
    with open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/normal_sentiment_words.txt", 'w', encoding='utf-8') as fp:
        for i in range(len(words)):
            fp.write(words[i] + ',' + str(weights[i]) + '\n')#将名字与权重写入


def process_words_list():
    sentences = get_data()
    words_list = []
    for sentence in sentences:
        words_list.extend(sentence)
    words_list = list(set(words_list))
    with open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/word_list.txt", 'w', encoding='utf-8') as fp:
        for word in words_list:
            fp.write(word + '\n')


def get_words_list():
    #process_words_list()
    words_list = []
    with open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/word_list.txt", 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            words_list.append(line.strip())
    return words_list


def get_stopwords():#将stopwords写入stopwords这个list中
    stopwords = []
    with open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/stop_words.txt", 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def get_sentiment_dict():
    sentiment_dict = {}
    with open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/normal_sentiment_words.txt", 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split(',')
            sentiment_dict[line[0]] = float(line[1])#建立dict
    return sentiment_dict


def process_word_vectors():
    from bert_serving.client import BertClient# Bert模型
    bc = BertClient()
    word_list = get_words_list()
    vecs = []
    for word in word_list:
        vec = bc.encode([word])
        vecs.append(vec[0])
    vecs = np.array(vecs)
    np.savetxt("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/vecs.txt", vecs)

def get_word_vectors():
    process_word_vectors()
    word_list = get_words_list()
    vecs = np.loadtxt("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/vecs.txt")
    word2vec = {}
    for i in range(len(word_list)):
        word2vec[word_list[i]] = vecs[i]
    return word2vec


def get_weighted_word_vectors():
    word2vec = get_word_vectors()
    sentiment_dict = get_sentiment_dict()
    for i in word2vec.keys():
        if i in sentiment_dict.keys():
            word2vec[i] = sentiment_dict[i] * word2vec[i]
    return word2vec


def get_data():
    stopwords = get_stopwords()
    sentiment_dict = get_sentiment_dict()
    sentences = []
    rs_sentences = []
    with open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/positive.txt", 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            sentences.append(line.strip())#将positive写入list sentences中
    with open("D:/data/sentimen-analysis-based-on-sentiment-lexicon-and-deep-learning-master/data/negative.txt", 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            sentences.append(line.strip())#将negative写入到list sentences中
            
    jieba.load_userdict(sentiment_dict.keys())#将sentiment_dict里面的词添加到jieba库里
    for sentence in sentences:
        sentence = list(jieba.cut(sentence))
        split_sentence = []
        for word in sentence:
            if '\u4e00' <= word <= '\u9fff' and word not in stopwords:#是否为中文并且是否不在stopwords里
                split_sentence.append(word)
        rs_sentences.append(split_sentence)
    return rs_sentences


def process_data(sentence_length, words_size, embed_size):
    #normalize_sentiment_words()
    #sentences = get_data()
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
        word_vectors[v, :] = torch.from_numpy(word2vec[k])
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
    label = [1 for _ in range(50000)]
    label.extend([0 for _ in range(50000)])
    label = np.array(label)
    return rs_sentences, label, word_vectors
