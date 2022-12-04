from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import CleanMessage as cleanMessage
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import accuracy_score, confusion_matrix

class NaiveBayes():
    def __init__(self):
        self.lastPreprop = []
        self.stop_factory = StopWordRemoverFactory()
        self.more_stopword = ['ibu', 'ayah', 'adik', 'kakak', 'nya', 'yah', 'sih', 'oke', 'kak', 'deh', 'mah', 'an', 'ku', 'mu', 'iya', 'apa',
                        'gapapa', 'akupun', 'apapun', 'eh', 'kah', 'mengada', 'apanya', 'tante', 'mas', 'suami', 'si', 'mama', 'bapak',
                        'nder', 'budhe', 'kakek', 'nenek', 'mbah', 'wow', 'kok', 'si', 'ke', 'ya', 'ohh', 'guyssssss','mjb', 'min', 'ya', 'to']
        with open('stopwords.txt') as f:
            for line in f:
                self.more_stopword.append(line.replace('\n',''))
        
        self.custom_stopword = self.stop_factory.get_stop_words() + self.more_stopword
        self.stopword = self.stop_factory.create_stop_word_remover()

        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()

        df = pd.read_csv('data_training/dt3b.csv')
        df['Label'].replace(to_replace='negative', value=0, inplace=True)
        df['Label'].replace(to_replace='positive', value=1, inplace=True)

        length = len(df)  # length of the datframe
        pos_count = len(df[df['Label'] == 1])  # positive_sentiment count
        neg_count = len(df[df['Label'] == 0])  # negative_sentiment count

        x = df['Text']
        y = df['Label']

        (n, p, bag_of_words) = self.bag_of_words_maker(x, y)

        prior_pos, prior_neg, table = self.naive_bayes_train(x, y)

        X = df["Text"]
        Y = df["Label"]
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state = 1)

        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        self.a, self.b, self.bag = self.naive_bayes_train(x_train, y_train)
        y_predicted = self.naive_bayes_predict(x_test,self.bag,self.a,self.b)[0]
        
        self.train_Accuracy= accuracy_score(y_test, np.array(y_predicted))
        (tn, fp, fn, tp) = confusion_matrix(y_test, np.array(y_predicted)).ravel()
        self.train_precsion= tp / (tp + fp)
        self.train_recall= tp / (tp + fn)
        self.train_fMeasure= 2 * ((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))
        self.train_errorRate= (fp + fn)/(tp + fp + tn + fn)

    # def getTrainAcc(self):
    #     return [(self.train_Accuracy), (self.train_precsion), (self.train_recall), (self.train_fMeasure), (self.train_errorRate)]

    def getTrainAcc(self):
        return ["{:.0%}".format(self.train_Accuracy), "{:.0%}".format(self.train_precsion), "{:.0%}".format(self.train_recall), "{:.0%}".format(self.train_fMeasure), "{:.0%}".format(self.train_errorRate)]

    # def getTrainAcc(self):
    #     return [round(self.train_Accuracy,2), round(self.train_precsion,2), round(self.train_recall,2), round(self.train_fMeasure,2), round(self.train_errorRate,2)]

    def parse_string(self, text):
        return ' '.join([str(item) for item in text])

    def n_gram(self, text, n: int = 2):
        txt = text.split( )
        result = list()  
        index = 0
        for t in txt:
            temp = []
            for i in range(n):
                if (index + i) >= len(txt):
                    break
                temp.append(txt[index + i])
            
            if len(temp) == n:
                string = ' '.join([str(item) for item in temp])
                result.append(string)
            index += 1                 
        return result

    def sentence_to_words(self, sentence, getData=False): #preprocessing
        preprop = []
        #clean
        sentence = cleanMessage.clean(sentence.lower())
        if(getData):
            preprop.append(sentence)
        spellingd = {}
        with open('spelling.csv') as f:
            for line in f:
                parts = line.split(',')

                parts[0] = parts[0].strip()
                parts[1] = parts[1].strip()
                
                spellingd[parts[0]] = parts[1]
                if not line:
                    break
        l = sentence.lower()  # convert sentence to lowercase
        if(getData): 
            preprop.append(l)
        #token
        l = l.split(' ')  # split sentence into individual word
        if(getData):
            preprop.append(l)
        p = ''
        word_list = []

        for word in l:
            p = ''
            for letter in word:
                if ord(letter) >= 67 and ord(letter) <= 122:
                    p = p + letter
            #spelling
            if p in spellingd:
                p = spellingd[p]
            word_list.append(p)
        if(getData):
            preprop.append(word_list)

            # print(word_list)
        #STOPWORD
        wordList = [word for word in word_list if word not in self.more_stopword]
        if(getData):
            preprop.append(wordList)
        # print(wordList)
        # return the word list of the sentence devoid of special characters and numericals

        #stemming
        stem_text = [self.stemmer.stem(word) for word in wordList]
        if(getData):
            preprop.append(stem_text)
        # print(stem_text)
        #ngram
        hasil_ngram = self.n_gram(self.parse_string(stem_text), 2)
        if(getData):self.lastPreprop.append(preprop)
        return hasil_ngram
    def getPreprop(self):
        return self.lastPreprop
    def naive_bayes_train(self, X, Y, a=0.000001):
        n_length = len(X)
        n_class_pos = len(Y[Y == 1])
        n_class_neg = len(Y[Y == 0])
        prior_pos = n_class_pos / n_length  # prior probability for  class
        prior_neg = n_class_neg / n_length  # prior probability for class
        # (n, p, bag) = self.bag_of_words_maker2(X, Y)
        (n, p, bag) = self.bag_of_words_maker(X, Y)

        pr = {}

        for i in range(len(bag)):  # evaluating the likelihood prob for each word given a class
            p_pos = (bag['count_pos'][i] + a) / (p + len(bag) * a)
            p_neg = (bag['count_neg'][i] + a) / (n + len(bag) * a)
            pr[bag['index'][i]] = [p_pos, p_neg]
        pr = pd.DataFrame(pr).T
        pr.columns = ['sent=positive', 'sent=negative']
        pr = pr.reset_index()
        return (prior_pos, prior_neg, pr)

    def naive_bayes_predict(self, 
        X,
        pr,
        prior_pos,
        prior_neg,
        getData=False
    ):
        Y = []
        dataDetail = []
        for i in range(len(X)):
            k_pos = 1
            k_neg = 1
            p = self.sentence_to_words(X[i], getData)

            for k in range(len(pr)):
                for word in p:
                    if word == pr['index'][k]:
                        # pdt of likelihood prob given the word is present in vocabulary
                        k_pos = k_pos * pr['sent=positive'][k]
                        k_neg = k_neg * pr['sent=negative'][k]
            # multiply each likelihood prob with the prior prob
            nb = [prior_neg * k_neg, prior_pos * k_pos]
            dataDetail.append([X[i], 'positif' if np.argmax(nb)==0 else 'negatif'])
            Y.append(np.argmax(nb))
        return [Y,dataDetail]

    def bag_of_words_maker(self, X, Y):

        bag_dict_binary_NB_pos = {}  # keeping track of the positive class words
        bag_dict_binary_NB_neg = {}  # keeping track of the negative class words
        stop_words = self.custom_stopword

        for i in range(len(X)):
            p = self.sentence_to_words(X[i])
            sent = Y[i]
            x_pos = {}
            # we intialize the dict every iteration so that it does not consider repititions .(Binary NB)
            x_neg = {}
            # print(p)
            if sent == 1:
                for word in p:
                    if word in x_pos.keys():
                        # word is the key and value stored is [count, sentiment]
                        x_pos[word] = [x_pos[word][0] + 1, x_pos[word][1]]
                    else:
                        x_pos[word] = [1, sent]
                for key in x_pos.keys():
                    if key in bag_dict_binary_NB_pos.keys():
                        bag_dict_binary_NB_pos[key] = \
                            [bag_dict_binary_NB_pos[key][0] + 1,bag_dict_binary_NB_pos[key][1]]
                    else:
                        # storing it in the final dict
                        bag_dict_binary_NB_pos[key] = [1, sent]
            if sent == 0:
                for word in p:
                    if word in x_neg.keys():
                        x_neg[word] = [x_neg[word][0] + 1, x_neg[word][1]]
                    else:
                        x_neg[word] = [1, sent]
                for key in x_neg.keys():
                    if key in bag_dict_binary_NB_neg.keys():
                        bag_dict_binary_NB_neg[key] = \
                            [bag_dict_binary_NB_neg[key][0] + 1,bag_dict_binary_NB_neg[key][1]]
                    else:
                        bag_dict_binary_NB_neg[key] = [1, sent]

        # print(bag_dict_multi.keys())
        # returns the dataframe containg word count in each sentiment
        neg_bag = pd.DataFrame(bag_dict_binary_NB_neg).T
        pos_bag = pd.DataFrame(bag_dict_binary_NB_pos).T

        neg_bag.columns = ['count_neg', 'sentiment_neg']
        pos_bag.columns = ['count_pos', 'sentiment_pos']
        try:
            neg_bag = neg_bag.drop(stop_words)
            pos_bag = pos_bag.drop(stop_words)
        except:
            print('None')
        neg_bag = neg_bag.reset_index()
        pos_bag = pos_bag.reset_index()
        n = len(neg_bag)
        p = len(pos_bag)
        bag_of_words = pd.merge(neg_bag, pos_bag, on=['index'], how='outer')
        bag_of_words['count_neg'] = bag_of_words['count_neg'].fillna(0)
        bag_of_words['count_pos'] = bag_of_words['count_pos'].fillna(0)
        bag_of_words['sentiment_neg'] = bag_of_words['sentiment_neg'].fillna(0)
        bag_of_words['sentiment_pos'] = bag_of_words['sentiment_pos'].fillna(1)
        return (n, p, bag_of_words)

    def predict(self, inputData):
        # self.lastPreprop = []
        y_predicted = self.naive_bayes_predict(inputData, self.bag, self.a, self.b, getData=True)
        total_pos = 0
        total_neg = 0
        for y_pred in y_predicted[0]:
            if y_pred == 1:
                total_pos += 1
            else:
                total_neg += 1
        return [(total_pos, total_neg), y_predicted[1]]