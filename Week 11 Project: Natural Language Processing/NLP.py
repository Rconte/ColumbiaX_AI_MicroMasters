import csv
import pandas as pd
import numpy as np
import time 
import os
import time 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier

#++++++++++++++++++++++++PREPROCESSING PHASE++++++++++++++++++++++++++++++++++++++++


def aid(list_of_replaced):

    proc = ' '.join([char for char in list_of_replaced if char.lower() not in stopwords])
    
    return proc

def preprocesstext(corpus):

    replaced = corpus.replace('<br />',' ')
    characters = ".,?!;:-_/<>@#%$^&*!(){}[]|"

    for character in characters:
        replaced = replaced.replace(character,'')
    replacelist = replaced.split()
    processed = aid(replacelist)
    
    return processed 

def grab_stopwords(filename):
    
    stops = open(filename,'r',encoding = 'ISO-8859-1').read()
    stops = stops.split('\n')
    print('GRABBED STOPWORDS..')
    return stops

def removestops(sent,stops):
    
    words = sent.split()
    result = [word for word in words if word.lower() not in stopwords]
    
    final = ' '.join(result)
   
    return final

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''process imdb data --- Implement this module to extract
    and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''

    global stopwords
    

    ind,text,rating = [],[],[]
    index =  0 
    stopwords = grab_stopwords('stopwords.en.txt')
    
    for filename in os.listdir(inpath+"pos"):
        
        csv_file = open(inpath+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
        csv_file = preprocesstext(csv_file)
        csv_file = removestops(csv_file, stopwords)
        
        
        ind.append(index)
        text.append(csv_file)
        rating.append("1")
        index+= 1

    for filename in os.listdir(inpath+"neg"):
        
        csv_file = open(inpath+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
        csv_file = preprocesstext(csv_file)
        csv_file = removestops(csv_file, stopwords)
        
        ind.append(index)
        text.append(csv_file)
        rating.append("0")
        index += 1

    print('PROCESSED TEXT BY REMOVING STOPWORDS,REMOVED SPACES AND CHARACTERS..')

    all_data = list(zip(ind,text,rating))
    
    if mix == True:
        print('MIXED')
        np.random.shuffle(all_data)
        
    else:
        print('NOT MIXED')
        pass

    dataframe = pd.DataFrame(data = all_data, columns=['row_Number', 'text', 'polarity'])
    dataframe.to_csv(outpath+name, index=False, header=True)


def grab_test_data(name= 'imdb_tr.csv'):
    '''get training data'''
    #grabs the test data from the imdb_tr csv file
    data = pd.read_csv(name,header = 0, encoding ='ISO-8859-1' )
    print('Test data snippet'.upper())
    print(data.head())
    
    X = data['text']
    print('SUCCESSFULLY GRABBED TEST DATA')
    
    return X

#grabs the training data from the imdb_tr csv file
def grab_train_data(name = 'imdb_tr.csv'):
    
    data = pd.read_csv(name,header = 0, encoding ='ISO-8859-1' )
    print('Data head snippet'.upper()) #sanity check
    print(data.head())
    X = data['text']
    Y = data['polarity']
    
    print('SUCCESSFULLY GRABBED TRAINING DATA')
    
    return X,Y   

def write(name, data):
    '''write text file '''

    with open(name,'w',newline = '') as output:
        filewriter = csv.writer(output)
        for result in data:
            filewriter.writerow([result])
    print('...')

#++++++++++++++++++++++++CHOOSE PROCESS TYPE (UNIGRAM,BIGRAM,TFIDF UNI OR BI)++++++++++++++++++++++++++++++++++++++++
#applies countvectorizer, checks whether we want to process data with unigram or bigram.
def process_data(data,process_type):
   
    '''Process unigram and bigram data'''

    #go for unigram
    
    if process_type == 'unigram':
        unigram_vectorizer = CountVectorizer()
        unigram_vectorizer = unigram_vectorizer.fit(data)
        print(process_type.upper(),'THROUGH..')
        #returns unigram vectorizer..
        
        return unigram_vectorizer
    
    #go for bigram
    elif process_type == 'bigram':
        ngram = 2
        bigram_vectorizer = CountVectorizer(ngram_range=(1,ngram))
        bigram_vectorizer = bigram_vectorizer.fit(data)
        print(process_type.upper(),'THROUGH..')
        #return bigram vectorizer..
        
        return bigram_vectorizer
  
#need to rework this one, it might be incorrect

def process_data_tfidf(data):
    '''Process data TFIDF'''

    #use transformer or vectorizer? vectorizer gives out an error when i try to process it in the same way   
    #Followup: since we are using the old test and training data, already processed through the process_data function,
    #we are only transforming the old data and using the tfidf transformer

    transformer = TfidfTransformer()
    transformer = transformer.fit(data)
    print('TFIDF PROCESSING THROUGH..')
    
    return transformer


#applies stochastic gradient descent
def SGD(X_train, Y_train, X_test):
    '''Stochastic gradient descend'''
    
    clf = SGDClassifier(loss="hinge", penalty="l1",max_iter = 10) #use penalty = l1 and loss = hinge
    clf.fit(X_train, Y_train)
    Y_test = clf.predict(X_test)
    print('SGD APPLICATION SUCCESSFUL')
    
    return Y_test


def main():

    global stopwords
    
    train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
    test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation

    print('FASE 1 - PREPROCESSING DATA FOR EVALUATION')
    
    imdb_data_preprocess(inpath=train_path, mix=False)
    [X_train, Y_train] = grab_train_data()
    X_test = grab_test_data(name=test_path)
    
    print('FASE 1 - COMPLETE')
    print('\n')
    print('FASE 2 - UNIGRAM PROCESSING')

    #train a SGD classifier using unigram representation,
    #predict sentiments on imdb_te.csv, and write output to
    #unigram.output.txt
    start = time.time()
    unigram_vectorizer = process_data(X_train,process_type = 'unigram')
    X_train_unigram = unigram_vectorizer.transform(X_train)
    X_test_unigram = unigram_vectorizer.transform(X_test)
    Y_test_unigram = SGD(X_train_unigram, Y_train, X_test_unigram)
    
    print('...WRITING....')
    write("unigram.output.txt",Y_test_unigram)
    end = time.time()
    
    print('FASE 2 - COMPLETE')
    print('elapsed : ', end - start)
    print('\n')
    print('FASE 3 - BIGRAM PROCESSING')
 
    #train a SGD classifier using bigram representation,
    #predict sentiments on imdb_te.csv, and write output to
    #bigram.output.txt
    start = time.time()
    bigram_vectorizer = process_data(X_train,process_type = 'bigram')
    X_train_bigram = bigram_vectorizer.transform(X_train)
    X_test_bigram = bigram_vectorizer.transform(X_test)
    Y_test_bigram = SGD(X_train_bigram, Y_train, X_test_bigram)
    
    print('...WRITING....')
    write("bigram.output.txt",Y_test_bigram)
    end = time.time()
    print('FASE 3 - COMPLETE')
    print('elapsed :', end -start)
    print('\n')
    print('FASE 4 - TFIDF UNIGRAM PROCESSING')
    
    #train a SGD classifier using unigram representation
    #with tf-idf, predict sentiments on imdb_te.csv, and write 
    #output to unigramtfidf.output.txt
    start = time.time()
    unigram_transformer_tifidf = process_data_tfidf(X_train_unigram)
    X_train_tfidf_unigram = unigram_transformer_tifidf.transform(X_train_unigram)
    #uses the unigram X test to obtain the tfidf version using transformer..
    X_test_tfidf_unigram = unigram_transformer_tifidf.transform(X_test_unigram)
    Y_test_tfidf_unigram = SGD(X_train_tfidf_unigram, Y_train, X_test_tfidf_unigram)
    
    print('...WRITING....')
    write("unigramtfidf.output.txt",Y_test_tfidf_unigram)
    end = time.time()
    print('FASE 4 - COMPLETE')
    print('elapsed:', end - start)
    print('\n')
    print('FASE 5 - TFIDF BIGRAM PROCESSING')
    

    #train a SGD classifier using bigram representation
    #with tf-idf, predict sentiments on imdb_te.csv, and write 
    #output to bigramtfidf.output.txt
    start = time.time()
    bigram_transformer_tfidf = process_data_tfidf(X_train_bigram)
    X_train_tfidf_bigram = bigram_transformer_tfidf.transform(X_train_bigram)
    #uses the bigram X test to obtain the tfidf version using transformer..
    X_test_tfidf_bigram = bigram_transformer_tfidf.transform(X_test_bigram)
    Y_test_tfidf_bigram = SGD(X_train_tfidf_bigram, Y_train, X_test_tfidf_bigram)
    
    print('...WRITING....')
    write("bigramtfidf.output.txt",Y_test_tfidf_bigram)
    end = time.time()
    print('FASE 5- COMPLETE')
    print('elapsed:', end-start)
    print('PROGRAM COMPLETE')
    
if __name__ == "__main__":
    total_start = time.time()
    print('program start'.upper())
    main()
    total_end = time.time()
    print('total time elapsed:', total_end - total_start)
    
    

