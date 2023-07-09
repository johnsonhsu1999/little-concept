import torch
import nltk


#---> torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = dim_of_word)
'''
假設vocab : {"I":0,"love":1,"eat":2,"you":3,"she":4,"he":5,"not":6}  --> size = 7
將2段句子轉成BOW的word list: [["you", "love", "he"],["I","not","eat","you"]] --> word list = [[3,1,5],[0,6,2,3]]
把這兩個句子透過embedding曾表示成每個字是dim=3, torch.nn.Embedding(8,3)其中表示：
1. 將size=8的dictionary
2. 每個word表示城dim=3的詞向量
'''

#example1 : 
#這裡因為要學習embedding層的內容所以才要同長度！所以每個句子一定要padding或減少到一定的長度才可以放入embedding層！！！
#但是！！！在訓練模型所放入的資料長度不一定要相等，透過DataLoader型別可以解決input句子長度不一的問題！！！！
vocab = {"I":0,"love":1,"eat":2,"you":3,"she":4,"he":5,"not":6} 
inputs = [["you love he"],["I not you"]] 

sents = [nltk.word_tokenize(str(sent[0])) for sent in inputs]
wordlist = []
for sent in sents:
    res = []
    for word in sent:
        res.append(vocab[word])
    wordlist.append(res)
wordlist = torch.tensor(wordlist,dtype=torch.long)  #轉成torch.tensor的格式！！！
embeddings = torch.nn.Embedding(num_embeddings=len(vocab),embedding_dim=4)
outputs = embeddings(wordlist)
print(f"word list : {wordlist}\n")  
print(f"outputs : {outputs}") 
#word list : tensor([[3, 1, 5],[0, 6, 3]])
#outputs : tensor([[[ 0.79, -0.41, -0.18,  1.52],...,[ 0.79, -0.49, -0.15,  1.52]]], grad_fn=<EmbeddingBackward0>)





#example2 : 
embedding = torch.nn.Embedding(7,3)  

inputs = torch.tensor([[0,1,2,1],[4,6,6,5]],dtype=torch.long) #inputs.shape = (2,4),共8個詞
output = embedding(inputs) #output.shape = (2,4,3)
print(output)



'''
1. 在一般的MLP訓練模型時，input的大小/長度可以不用一致，因為可以透過紀錄句子的長度offset，來放進模型裡做訓練，
   主要是因為每個句子可能不等長，所以有解決的機制

2. 但是在CNN卷積神經網路之下，input大小要一致。
   因為通常CNN是要算圖像為主的問題，所以在文件處理上也要對句子補齊或修短。
   補齊方法為: torch.nn.utils.rnn.pad_sequence，機制為句子最大長度，然後不足補0 ---> 句子的type一定要是tensor

   from torch.nn.utils.rnn import pad_sequence
   lis = [[1,3,4,6,2],[5,2,1],[2,2],[5,6,7,3,7],[8],[6,6,2,6,1],[2,4,3]]
   inputs = [torch.tensor(sent) for sent in lis] 
   outputs = pad_sequence(inputs, batch_first=True)
   print(outputs) 

   #--->tensor([[1, 3, 4, 6, 2],
        [5, 2, 1, 0, 0],
        [2, 2, 0, 0, 0],
        [5, 6, 7, 3, 7],
        [8, 0, 0, 0, 0],
        [6, 6, 2, 6, 1],
        [2, 4, 3, 0, 0]])




'''
