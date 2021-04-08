import torch as t
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data
from data_loader import MyData
from model import SLCABG
from test1 import *
import matplotlib.pyplot as plt

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')# 把所建立的模型全部迁移到 GPU
SENTENCE_LENGTH = 20
WORD_SIZE = 35000
EMBED_SIZE = 200
epochs = 15

if __name__ == '__main__':
    #process_sentiment_words()
    ##normalize_sentiment_words()
    #process_words_list()
    #process_word_vectors()
    
    #normalize_sentiment_words()
    #process_sentiment_words()
    
    acc_matrix = pd.DataFrame(None,columns=['acc'])
    r_matrix = pd.DataFrame(None,columns=['r'])
    f1_matrix = pd.DataFrame(None,columns=['f1'])
    
    sentences, label, word_vectors = data_util.process_data(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)
    x_train, x_test, y_train, y_test = train_test_split(sentences, label, test_size=0.2)

    train_data_loader = torch.utils.data.DataLoader(MyData(x_train, y_train), 32, True)
    test_data_loader = torch.utils.data.DataLoader(MyData(x_test, y_test), 32, False)

    net = SLCABG(EMBED_SIZE, SENTENCE_LENGTH, word_vectors).to(device)#发送到GPU中
    
    # construct loss and optimizer
    optimizer = t.optim.Adam(net.parameters(), 0.01)
    criterion = nn.CrossEntropyLoss()
    tp = 1
    tn = 1
    fp = 1
    fn = 1
    for epoch in range(epochs):
        for i, (cls, sentences) in enumerate(train_data_loader):
            optimizer.zero_grad()#梯度归零
            sentences = sentences.type(t.LongTensor).to(device)#to(device)转移到GPU中
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * r * p / (r + p)
            acc = (tp + tn) / (tp + tn + fp + fn)
            loss = criterion(out, cls).to(device)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                acc_matrix = acc_matrix.append({'acc':acc},ignore_index = True)
                r_matrix = r_matrix.append({'r':r},ignore_index = True)
                f1_matrix = f1_matrix.append({'f1':f1},ignore_index = True)
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                print('acc', acc, 'p', p, 'r', r, 'f1', f1)

    net.eval()
    print('==========================================================================================')
    with torch.no_grad():
        tp = 1
        tn = 1
        fp = 1
        fn = 1
        for cls, sentences in test_data_loader:
            sentences = sentences.type(t.LongTensor).to(device)
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * r * p / (r + p)
        acc = (tp + tn) / (tp + tn + fp + fn)
        print('acc', acc, 'p', p, 'r', r, 'f1', f1)



export_path = 'C:/Users/徐泽玮/Desktop/Patrick program/week7/dataset/'
new_acc = make_list(acc_matrix,'acc',epochs)
dump_object(new_acc,"new_acc", export_path)
new_r = make_list(r_matrix,'r',epochs)
dump_object(new_r,"new_r", export_path)
new_f1 = make_list(f1_matrix,'f1',epochs)
dump_object(new_f1,"new_f1", export_path)

#画图
x = np.linspace(0, epochs *411, epochs)
plt.figure()    # 定义一个图像窗口
plt.subplot(2,2,1)
y1 = new_acc
#y1 = acc_matrix.to_numpy()
plt.plot(x, y1.T) # 绘制曲线 y1
plt.subplot(2,2,2)
y2 = new_r
#y2 = r_matrix.to_numpy()
plt.plot(x, y2) # 绘制曲线 y2
plt.subplot(2,1,2)
y3 = new_f1
#y3 = f1_matrix.to_numpy()
plt.plot(x, y3) # 绘制曲线 y3
plt.show()
