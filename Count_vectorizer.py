'''
word embedding:
1. Bag of words-->詞語袋的概念，n個詞語就有n維向量，一個句子中出現那個單字就標1，反之則標0
   ---->實現方法為用sklearn 的 CountVectorizer

2. 進化版為tf-idf, 因為countervectorizer將每個單字重要性都作為相等，
   tf:term freq單字出現次數,  idf:inverse document freq有出現此單字的文句數量 -->相乘
   但如果幾篇文章出現很多“因為、所以、然後...“這種，不應該視為內容相似
   例如”因為“這個詞在80%的文句裡都出現-->所以重要性低
   然而“匯率”這個詞在2%的文句出現-->代表重要程度較高
   ---->實現方法為用sklearn 的 tfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() / TfidfVectorizer()...
data = cv.fit_transform(文句s串列)
feature_name = cv.get_feature_names_out()
array = data.toarray()
'''


#1. CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
texts = ['This is a first document',
         'This  document is a second document',
         'And this is a third one',
         'Is a first document']
#(1)建立vectorizer物件
vectorizer  = CountVectorizer()   #為Bag Of Words的實現，回傳物件 one of the word embedding method
#(2)將資料放進vectorizer物件
word_vectors = vectorizer.fit_transform(texts)
#(3)顯示特徵內容、矩陣表示
  #print(f"word_vector物件:{word_vectors}")
print(f"Features : {vectorizer.get_feature_names_out()}")  #找出所有的特徵、字典內容
print(f"Embedded value by count: \n{word_vectors.toarray()}")  #要用“.toarray"才能真正表示 


import pandas as pd
dataset = pd.DataFrame({value : word_vectors.toarray()[:,index] for index, value in enumerate(vectorizer.get_feature_names_out())})
print(dataset,'\n\n')

#2. TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer2 = TfidfVectorizer()
texts = ['This is the first document',
         'This  document is the second document',
         'And this is the third one',
         'Is the first document']

data = vectorizer2.fit_transform(texts)
to_array = data.toarray()
features = vectorizer2.get_feature_names_out()
print("Embedded value by tfidf : \n",to_array)
print("Features : \n",features)






#3 pmi,ppmi作為tf-idf的概念
#分散語意word embedding
'''
sent1 = 我 喜歡 自然 語言 處理。
sent2 = 我 愛 深度 學習。
sent3 = 我 喜歡 機器 學習。
以下矩陣表示詞語共現頻次表，也就是dict[vocab][vocab]的矩陣，當某詞上下文出現過另外一個詞則+1

     我 喜歡 自然 語言 處理 愛 深度 學習 機器 。
我   0   2   1   1    1   1  1   2    1  3
喜歡 2   0   1   1    1   0  0   1    1  2
自然
語言
處理
愛
深度
學習
機器

'''
#pmi calculate
import numpy as np
M = np.array([[0,2,1,1,1,1,1,2,1,3],
              [2,0,1,1,1,0,0,1,1,2],
              [1,1,0,1,1,0,0,0,0,1],
              [1,1,1,0,1,0,0,0,0,1],
              [1,1,1,1,0,0,0,0,0,1],
              [1,0,0,0,0,0,1,1,0,1],
              [1,0,0,0,0,1,0,1,0,1],
              [2,1,0,0,0,1,1,0,1,2],
              [1,1,0,0,0,0,0,1,0,1],
              [3,2,1,1,1,1,1,2,1,0]])

def pmi(matrix, positive=True):  #by PMI及MLE公式
    col_totals = matrix.sum(axis=0) #col sum
    row_totals = matrix.sum(axis=1) #row sum
    total = matrix.sum()
    expected =  np.outer(row_totals,col_totals)/total  #outer(a,b) = b transpose x a --> n*n matrix
    matrix = matrix/expected  #矩陣的個別元素相除

    with np.errstate(divide='ignore'):  #這裡因為不讓加權值變負數，則用np的警告，有點像try and error
        matrix = np.log(matrix)
    if positive:
        matrix[matrix<0] = 0.0

    return matrix

M_pmi = pmi(matrix=M)
np.set_printoptions(precision=2)  #列印結果保留兩位小數
print(M_pmi)