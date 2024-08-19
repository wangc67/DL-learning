import torch
import numpy as np
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from collections import Counter, OrderedDict
from torch.nn.utils.rnn import pad_sequence
import re
from gensim.models import Word2Vec
'''
https://github.com/Loche2/IMDB_RNN/blob/master/training.py
https://blog.csdn.net/nlpuser/article/details/88067167
https://blog.csdn.net/qq_60587058/article/details/131751967
https://blog.csdn.net/u010442263/article/details/130036050

https://blog.csdn.net/m0_64336780/article/details/127906964

useful
'''

def GetIMDB(batch_size: int):
    train_iter, test_iter = IMDB(split=('train', 'test'))
    tokenizer = get_tokenizer('basic_english')
    counter = Counter() # Counter() 主要功能：可以支持方便、快速的计数，将元素数量统计，然后计数并返回一个字典，键为元素，值为元素个数。
    for label, line in train_iter:
        counter.update(tokenizer(line))
    sorted_counter = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    vocabulary = vocab(sorted_counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
    vocabulary.set_default_index(vocabulary['<unk>'])

    def collate_batch(batch): # collate_fn的用处:自定义数据堆叠过程,自定义batch数据的输出形式
        label_list, text_list = [], []
        text_transform = lambda x: [vocabulary['<BOS>']] + [vocabulary[token] for token in tokenizer(x)] + [vocabulary['<EOS>']]
        label_transform = lambda x: x % 2
        for (_label, _text) in batch:
            label_list.append(label_transform(_label))
            processed_text = torch.tensor(text_transform(_text))
            text_list.append(processed_text)
        return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

    TrainLoader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    TestLoader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return TrainLoader, TestLoader

path = './data/emotions/'

class dataset():
    def __init__(self, file, maxlen=40) -> None:
        self.maxlen = maxlen
        self.vector_size = 20

        with open(file, 'r') as f:
            lst = re.split(';|\n', f.read())
        self.sentences = lst[::2]
        self.labels = lst[1::2]
        # emotion_set = set(labels)
        # print(emotion_set) 
        # {'fear', 'anger', 'joy', 'love', 'sadness', 'surprise'}
        bijection = {'fear': 0, 'anger': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5}
        for i in range(len(self.labels)):
            self.labels[i] = bijection[self.labels[i]]
        # print(type(self.labels), type(self.labels[0]))
        # input()
        tokenizer = get_tokenizer('basic_english')    
        for i in range(len(self.sentences)):
            self.sentences[i] = tokenizer(self.sentences[i])
        self.word2vec = Word2Vec(self.sentences, vector_size=self.vector_size, window=3, min_count=1, epochs=10, negative=0, hs=1) # 
        # window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值, neg=0: dont use negative sampling
        # When both 'hs=0' and 'negative=0', there will be no training.
        for i in range(len(self.sentences)):
            self.sentences[i] = [self.word2vec.wv.get_vector(word) for word in self.sentences[i]]
            if len(self.sentences[i]) >= self.maxlen:
                self.sentences[i] = self.sentences[i][0: self.maxlen - 1]
            else:
                self.sentences[i].extend([[0 for _ in range(self.vector_size)] for __ in range(self.maxlen - len(self.sentences[i]))])
            self.sentences[i] = torch.tensor(np.array(self.sentences[i]))
        self.len = len(self.labels)
        print('dataset initialized') # ---------------------------------------------------

    def __getitem__(self, index):
        return (self.sentences[index], self.labels[index])

    def __len__(self):
        return self.len
    
def GetDataLoader(batch_size: int, mode='train'):
    '''
    mode == 'train': return train, val;
    else return test
    '''
    if mode == 'train':
        TrainDataset = dataset(path + 'train.txt')
        ValDataset = dataset(path + 'val.txt')
        TrainLoader = DataLoader(TrainDataset, batch_size, shuffle=True, pin_memory=True)
        ValLoader = DataLoader(ValDataset, batch_size, shuffle=False, pin_memory=True)
        return TrainLoader, ValLoader
    else:
        TestDataset = dataset(path + 'test.txt')
        TestLoader = DataLoader(TestDataset, batch_size, shuffle=False, pin_memory=True)
        return TestLoader

if __name__ == '__main__':
    print('NLP_data_loader.py')
    '''
    import matplotlib.pyplot as plt
    def get_len_emo_maxlen(file):
        with open(file, 'r') as f:
            lst = re.split(';|\n', f.read())
        sentences = lst[::2]
        labels = lst[1::2]
        bijection = {'fear': 0, 'anger': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5}
        emo = [0 for _ in range(6)]
        for i in range(len(labels)):
            labels[i] = bijection[labels[i]]
            emo[labels[i]] += 1
        tokenizer = get_tokenizer('basic_english')  
        maxlen = 0
        length = []
        for i in range(len(sentences)):
            sentences[i] = tokenizer(sentences[i])
            maxlen = max(maxlen, len(sentences[i]))
            length.append(len(sentences[i]))
        return length, emo, maxlen
    len1, emo1, maxlen1 = get_len_emo_maxlen(path+'train.txt')
    len2, emo2, maxlen2 = get_len_emo_maxlen(path+'val.txt')
    len3, emo3, maxlen3 = get_len_emo_maxlen(path+'test.txt')
    emo = [emo1[i] + emo2[i] + emo3[i] for i in range(6)]
    maxlen = max(maxlen1, maxlen2, maxlen3)
    print(emo1)
    print(emo2)
    print(emo3)
    print(emo)
    print(maxlen)
    length = []
    length.extend(len1)
    length.extend(len2)
    length.extend(len3)
    print(len(length), 'asdfaf')
    plt.hist(length, bins=30)
    plt.show()
    '''
    
    '''
    test_loader = GetDataLoader(mode='test', batch_size=8)
    for line, label in test_loader:
        print(type(line), type(label))
        print(line, label)
        print(line.shape, label.shape)
        exit()

    train_dataloader, _ = GetIMDB(12)
    for idx, (label, text) in enumerate(train_dataloader):
        print(idx, label, text)
        print(label.shape, text.shape)
        exit()
    '''
    
