'''
word vector可用共現矩陣表示，效果比bow的獨熱向量來得好
訓練好的共現矩陣，每一行代表一個詞的特徵list,word vector
1. make a freq dictionary --> encode_dictionary(sentences)
2. make Co Occurrence Matrices --> co_occur_matrix(sentences, dic):
'''

import numpy as np
import nltk
from nltk.corpus import stopwords
Stopwords = set(stopwords.words('english'))


def encode_dictionary(sentences): #sort by freq from high to low
    #(1)Preprocessing : 分詞，要先化小寫、句子合併、去除nltk的stopwords(這裡應該要先用正規表達式來去除標點符號那類的東西)
    sentences = ' '.join(sentences)
    sentences = nltk.word_tokenize(sentences.lower()) 
    tokens = []
    for word  in sentences: 
        if (word in Stopwords) or (word=='.') or (word==','):
            continue
        else:
            tokens.append(word)

    #(2)make dictionary 並用value(freq大到小來排序)來排序字典！！！要用lambda
    D = dict()
    for token in tokens:
        if token not in D:
            D[token]=1
        else:
            D[token]+=1
    D = sorted(D.items(),reverse=True,key=lambda x:x[1]) 

    #(3) 依照上面求得的freq高低來編碼
    index = 0
    Dict = dict()
    for i in range(len(D)):
        Dict[D[i][0]]=index
        index+=1
    return Dict



def co_occur_matrix(sentences, dic):
    #(1)matrix preprocess
    size = len(dic)
    M = np.zeros(shape=(size,size))

    #(2)共現矩陣放入值
    for sent in sentences:
        sent = nltk.word_tokenize(sent.lower())
        for i in range(len(sent)):
            for j in range(len(sent)):
                if sent[i] in dic and sent[j] in dic:
                    M[dic[sent[i]]][dic[sent[j]]]+=1
    #(3)對角為0
    for i in range(size):
        for j in range(size):
            if i==j:   
                M[i][j]=0
    return M



#Example : 
texts = ["Tom is a king, but Amy is a king, too.","Amy is a queen.","King should be Tom, not Amy.","Tom loves Amy."]
Dictionary = encode_dictionary(texts)
print("dictionary :",Dictionary)
matrix = co_occur_matrix(sentences=texts, dic = Dictionary)
print("final\n",matrix)


