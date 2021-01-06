#!/usr/bin/env python
# coding: utf-8

# In[7]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


data_path= r"/content/drive/MyDrive/IMDB Dataset.csv""


# In[4]:


import nltk 
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize,sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import pprint
import matplotlib.pyplot as plt
from nltk import bigrams 
from nltk.corpus import sentiwordnet as swn
from sklearn import metrics
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')


# 
# 
# # Q1

# In[8]:


#Q1 - supervised learning

#read data
def read_data(data_path):
    #read data
    df = pd.read_csv(data_path)
    
    #split data by train and test
    train, test = train_test_split(df, test_size=0.5)
    tmp_train = train.head(10)
    tmp_test= test.head(10)
    
    return train, test,tmp_train, tmp_test


# In[9]:




def pre_processing(df):

    #tokenize
    data_dict= {}
    print("df ", len(df))
    for row in df.values:
        tokenize_text = word_tokenize(row[0])   
        data_dict[tuple(tokenize_text)] = row[1]
    print("data_dict ", len(data_dict.items()))

    #convert to lower case
    data_dict_after_lower={}
    for content,label in data_dict.items():
        lower_content= list(map(lambda word:word.lower(),content))
        data_dict_after_lower[tuple(lower_content)]= label
    print("data_dict_after_lower ", len(data_dict_after_lower.items()))

    #stopwords removel
    data_dict_no_stop_words={}
    stop_words = set(stopwords.words('english'))
    for word in [".",",","(",")","<",">","br","!","/","--","n't","'s","''","?","...","``",":","-","'","would",";","*"]:
        stop_words.add(word)
    for content,label in data_dict_after_lower.items():
        filtered_content = [w for w in content if not w in stop_words]
        data_dict_no_stop_words[tuple(filtered_content)]= label
    print("data_dict_no_stop_words ", len(data_dict_no_stop_words.items()))
      
    #lemmatization
    wnl = nltk.WordNetLemmatizer()
    data_dict_after_lemmatization={}
    for content,label in data_dict_no_stop_words.items():
        lemmatize_content = [wnl.lemmatize(w) for w in content]
        data_dict_after_lemmatization[tuple(lemmatize_content)]= label 
    print("data_dict_after_lemmatize ", len(data_dict_after_lemmatization.items()))
    
    return data_dict_after_lemmatization


def neg_pos_graph(terms_num):
    plt.bar(list(range(1, 3)), terms_num, tick_label=['positive', 'negative'],
            width=0.8, color=['red', 'yellow'])
    plt.xlabel('Category')
    plt.ylabel('#terms')
    plt.title('Terms Distribution Per Category')
    plt.show()


def explore_data(prepared_tmp):
    count_pos=0
    count_neg=0
    pos_terms=[]
    neg_terms=[]
    for content,label in prepared_tmp.items():
        if(label=='positive'):
            count_pos+=1
            pos_terms.extend(content)
        else:
            count_neg+=1
            neg_terms.extend(content)
    print("positive number is ",count_pos)
    print("negative number is ",count_neg)
    neg_pos_graph([count_pos, count_neg])
    
    def calculate_distribution(cur_count, catrgory):
        occ_nums={}
        for occ_num in cur_count.values():
            if occ_num in occ_nums:
                occ_nums[occ_num]= occ_nums[occ_num]+1
            else:
                 occ_nums[occ_num]=1
        final_array={}
        for occ_num, occ in occ_nums.items():
            if occ_num>5 and occ_num<8000: 
                final_array[occ_num]=occ
        plt.bar( occ_nums.values(),occ_nums.keys(), color='green', width=0.8)
        plt.xlabel("the number of terms" )
        plt.ylabel('number of occurrences')
        plt.title(str(catrgory)+ " distribution graph")
        axes = plt.gca()
        axes.set_ylim([0,250])
        axes.set_xlim([0,400])
        plt.show()
        

    count_pos= Counter()
    count_pos.update(pos_terms)
    pp_pos = pprint.PrettyPrinter()
    print("positive catrgory most commons terms: ")
    pp_pos.pprint(count_pos.most_common(10))
    calculate_distribution(count_pos, "positive")
    
    count_neg= Counter()
    count_neg.update(neg_terms)
    pp_neg = pprint.PrettyPrinter()
    print("negative catrgory most commons terms: ")
    pp_neg.pprint(count_neg.most_common(10))
    calculate_distribution(count_neg, "negative")
                
    bigram= bigrams(pos_terms)
    count_bigram = Counter()
    count_bigram.update(bigram)
    print("bigram of positive catrgory: \n")
    print(count_bigram.most_common(10))
    calculate_distribution(count_bigram, "positive")


    bigram= bigrams(neg_terms)
    count_bigram = Counter()
    count_bigram.update(bigram)
    print("bigram of negative catrgory: \n")
    print(count_bigram.most_common(10))
    calculate_distribution(count_bigram, "negative")


    
train, test, tmp_train, tmp_test= read_data(data_path)

prepared_train= pre_processing(train)  
prepared_test= pre_processing(test) 
# prepared__train_tmp= pre_processing(tmp_train)
# prepared__test_tmp=pre_processing(tmp_test)

explore_data(prepared_train) # we sould insert prepared_train


# In[10]:


#Q2 -  un-supervised learning

#read data
# df = pd.read_csv(data_path)

def pre_processing(df):  
    #tokenize
    data= []
    for row in df.values:
        tokenize_text = nltk.word_tokenize(row[0])
        data.append(tokenize_text)
     
    #convert to lower case
    data_after_lower=[]
    for content in data:
        lower_content= list(map(lambda word:word.lower(),content))
        data_after_lower.append(lower_content)

    #stopwords removel
    data_without_stop_words=[]
    stop_words = set(stopwords.words('english'))
    for word in [".",",","(",")","<",">","br","!","/","--","n't","'s","''","?","...","``",":","-","'","would",";","*"]:
        stop_words.add(word)
    for content in data_after_lower:  
        filtered_content = [w for w in content if not w in stop_words] 
        data_without_stop_words.append(filtered_content)
    
    #lemmatization
    wnl = nltk.WordNetLemmatizer()
    clean_data = []
    for content in data_without_stop_words:
        lemmatize_content = [wnl.lemmatize(w) for w in content]
        clean_data.append(lemmatize_content) 
    
    return clean_data

def classify_data(clean_data):
    tagged_list=[]
    final_docs_score = []
    score_list = []
    
    #Create POS tagging for each token in each doc    
    tagged_list=[]
    for content in clean_data:
        tagged_list.append(nltk.pos_tag(content)) 
        
    for idx,doc in enumerate(tagged_list):  
        score_list.append([]) 
        for idx2,t in enumerate(doc):  #t[0] word, t[1] pos tag
            newtag=''
            if t[1].startswith('NN'):
                newtag='n'
            elif t[1].startswith('JJ'):
                newtag='a'
            elif t[1].startswith('V'):
                newtag='v'
            elif t[1].startswith('R'):
                newtag='r'
            else:
                newtag=''       
            if(newtag!=''):   
                synsets = list(swn.senti_synsets(t[0], newtag))
                score=0
                if(len(synsets)>0):
                    for syn in synsets:
                        score+=syn.pos_score()-syn.neg_score()
                    score_list[idx].append(score/len(synsets)) #add score of each term in doc
        
    # Create final score to each doc(positive or negative)   
    for score_sent in score_list:
        final_docs_score.append(sum([word_score for word_score in score_sent])/len(score_sent))
    
    #craete dict -> key: doc content, value: class(positive or negative)
    labled_dict = {}
    for idx,doc in enumerate(clean_data):       
        if final_docs_score[idx] > 0:
            value = "positive"
        elif final_docs_score[idx] < 0:
            value = "negative"
        else: #objective case
            continue
        labled_dict [tuple(doc)] = value   
   
    return labled_dict, final_docs_score

def calc_accuracy(docs_score):
    predict_lables = []
    for score in docs_score:
        if score > 0:
            lable = "positive"
        elif score < 0:
            lable = "negative"
        else: #objective case
            continue 
        predict_lables.append(lable)
    
    acctual_lables=[]
    for row in test.values:
        acctual_lables.append(row[1])
        
    score = metrics.accuracy_score(acctual_lables, predict_lables)
    print("accuracy:",score)
    
#call functions
train, test,tmp_train, tmp_test= read_data(data_path)
clean_data = pre_processing(test)
labled_dict_data,docs_score = classify_data(clean_data)
calc_accuracy(docs_score)
explore_data(labled_dict_data)


# In[ ]:


# Q3

# content of the train data
content_train=[]
for index in range(len(prepared_train.keys())):
    content_train.append(' '.join(list(prepared_train.keys())[index]))

# target of the train data
target_train= list(prepared_train.values())

# content of the test data
content_test= []
for index in range(len(prepared_test.keys())):
    content_test.append(' '.join(list(prepared_test.keys())[index]))
    
# target of the test data
target_test= list(prepared_test.values())



# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf():
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

    train = vectorizer.fit_transform(content_train)
    test = vectorizer.transform(content_test)

    # print(vectorizer.get_feature_names())

    print("n_samples: %d, n_features: %d" % train.shape)
    print()
    return train, test


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words():
    vectorizer = CountVectorizer()
    train = vectorizer.fit_transform(content_train)
    test = vectorizer.transform(content_test)

    print("n_samples: %d, n_features: %d" % train.shape)
    print()
    return train, test


# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB


# In[ ]:


from sklearn import metrics

# run the supervised method- clf on the train data and predict the test data
def benchmark(clf, train, test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    clf.fit(train, target_train)
    pred = clf.predict(test)
    score = metrics.accuracy_score(target_test, pred)
    print("accuracy:   %0.2f" % score)
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score


# In[ ]:


import numpy as np

# Before tuning the parameters, run the diffrent supervised learning models with the diffrent feature extract methods
results = []
for feature_extraction in ['tf_idf', 'bag_of_words']:
    if feature_extraction == 'tf_idf':
         train, test= tf_idf()
    else:
        train, test=bag_of_words()
    for clf, name in (
            (SGDClassifier(),"SVM"),
            (Perceptron(), "Perceptron"),
            (MultinomialNB(),"Naive Bayes")):
        print('=' * 80)
        print('machine learning methods:',name, ', feature_extraction:', feature_extraction )
        results.append(benchmark(clf,train, test))



# # <h3> Optimize the Classifiers </h3>
# 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 

# Tuning the parameters of the diffrent supervised learning models with all kinds of the diffrent feature extract method
for clf, method_name in (
            (MultinomialNB(),"Perceptron"),
            (SGDClassifier(),"SVM"),
            (Perceptron(), "Naive Bayes")):
    for vect, feature_ex_name in (
            (TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english'),"tf_idf"),
            (CountVectorizer(),"Bag of words")):
        print('clf : ',method_name, ', vect' ,feature_ex_name )
        nb_clf = Pipeline([('vect', vect),('clf', clf)])
        parameters={}
        if method_name=="Naive Bayes":
            parameters =  {'vect__max_df': (0.3,0.5),'clf__alpha': (0.01,1.0)}
            gs_clf = GridSearchCV(nb_clf, parameters, n_jobs=1)
            gs_clf = gs_clf.fit(content_train,target_train)

        if method_name=="SVM":
            parameters = { 'vect__max_df': (0.3,0.5),'clf__alpha': (0.0001,0.01,1.0), 'clf__n_iter_no_change': (5, 10), 'clf__max_iter': (1000,1020)}
            gs_clf = GridSearchCV(nb_clf, parameters, scoring='accuracy', cv=5)
#             for param in gs_clf.get_params().keys(): 
#                 print(param)
            gs_clf = gs_clf.fit(content_train,target_train)
            
        if method_name=="Perceptron":
            parameters =  {'vect__max_df': (0.3,0.6),'clf__alpha': (1.0,1.4)}
            gs_clf = GridSearchCV(nb_clf, parameters, scoring='accuracy', cv=5)
#             for param in gs_clf.get_params().keys(): 
#                 print(param)
            gs_clf = gs_clf.fit(content_train,target_train)
            
        
        predictions =gs_clf.predict(content_test)      
        print(classification_report(target_test, predictions))
        
        print('Best score:  %0.2f ' % gs_clf.best_score_)
        print('Best params: ',gs_clf.best_params_)
        print('=' * 80)

        


# In[ ]:




