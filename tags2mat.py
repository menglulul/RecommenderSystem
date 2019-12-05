#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:58:29 2019

@author: lulumeng
"""

def load_tags(Path):
    return tag_data

def gen_vocab:
    K = 100 # choose top K words
    dict = {}
    return vocab


    K = 100 # choose top K words
    negpos = ["neg/","pos/"]
    dict = {}
    Xtrain_text = []
    Xtrain = []
    Xtest = []
    ytrain = []
    ytest = []
    porter=PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    #training vocabulary
    for i in range(2):
        fpath = Path+"training_set/"+negpos[i]
        files = os.listdir(fpath)
        for file in files:
            ytrain.append(i)
            f = open(fpath+file, 'r')
            sentences = f.read()
            f.close()
            words = tokenizer.tokenize(sentences)
            stemmed = []
            for w in words:
                stemmed.append(porter.stem(w))
            Xtrain_text.append(stemmed)
    
    for text in Xtrain_text:
        for word in text:
            if word in dict:
                dict[word] += 1
            else:
                dict[word] = 1
    vocab = []
    stop_words = set(stopwords.words('english')) 
    sorted_dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    cnt=0
    for item in sorted_dict:
        if item[0] not in stop_words:
            vocab.append(item[0])
            cnt +=1
        if K==cnt:
            break
    print(vocab)
    
    for text in Xtrain_text:
        Xtrain.append(transfer_method2(text, vocab))
        
    for i in range(2):
        fpath = Path+"test_set/"+negpos[i]
        files = os.listdir(Path+"test_set/"+negpos[i])
        for file in files:
            ytest.append(i)
            f = open(fpath+file, 'r')
            words = f.read().split()
            f.close()
            stemmed = []
            for w in words:
                stemmed.append(porter.stem(w))
            Xtest.append(transfer_method2(stemmed, vocab))
    return Xtrain, Xtest, ytrain, ytest
