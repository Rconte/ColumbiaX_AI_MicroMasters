import csv
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier


    
train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):

    global stopwords
    stopwords = grab_stopwords('stopwords.en.txt')

    ind,text,rating = [],[],[]
    index =  0 

    for filename in os.listdir(inpath+"pos"):
        data = open(inpath+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
        data = preprocesstext(data)
        data = removestops(data, stopwords)
        
        ind.append(index)
        text.append(data)
        rating.append("1")
        index+= 1

    for filename in os.listdir(inpath+"neg"):
        data = open(inpath+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
        data = preprocesstext(data)
        data = removestops(data, stopwords)
        ind.append(index)
        text.append(data)
        rating.append("0")
        index += 1

    Dataset = list(zip(ind,text,rating))
    
    if mix:
        np.random.shuffle(Dataset)

    df = pd.DataFrame(data = Dataset, columns=['row_Number', 'text', 'polarity'])
    df.to_csv(outpath+name, index=False, header=True)

#try with and without

def preprocesstext(corpus):

    replaced = corpus.replace('<br />',' ')
    characters = ".,?!;:-_/<>@#%$^&*!(){}[]|"

    for character in characters:
        replaced = replaced.replace(character,'')
    replacelist = replaced.split()
    processed = ' '.join([char for char in replacelist if char.lower() not in stopwords])
    #processed = procs(replacelist,stopwords)
    return processed 

def grab_stopwords(filename):
    
    stops = open(filename,'r',encoding = 'ISO-8859-1').read()
    stops = stops.split('\n')
    return stops

def removestops(sent,stops):
    
    words = sent.split()
    result = [word for word in words if word.lower() not in stopwords]
    final = ' '.join(result)
    return final

    
def process_data(data,process_type):
    
    if process_type == 'unigram':
        unigram_vectorizer = CountVectorizer()
        unigram_vectorizer = unigram_vectorizer.fit(data)
        print(process_type)
        return unigram_vectorizer
    elif process_type == 'bigram':
        ngram = 2
        bigram_vectorizer = CountVectorizer(ngram_range=(1,ngram))
        bigram_vectorizer = bigram_vectorizer.fit(data)
        print(process_type)
        return bigram_vectorizer
  

def process_data_tfidf(data, process_type):
    if process_type == 'unigram':
        ngram = 1
        transformer = TfidfTransformer()
        transformer = transformer.fit(data)
        return transformer
        
    elif process_type == 'bigram':
        ngram = 1
        transformer = TfidfTransformer()
        transformer = transformer.fit(data)
        return transformer

def grab_test_data(name= 'imdb_tr.csv'):
    
    data = pd.read_csv(name,header = 0, encoding ='ISO-8859-1' )
    X = data['text']
    return X

def grab_train_data(name = 'imdb_tr.csv'):
    
    data = pd.read_csv(name,header = 0, encoding ='ISO-8859-1' )
    X = data['text']
    Y = data['polarity']
    return X,Y   


def stochastic_descent(Xtrain, Ytrain, Xtest):
    from sklearn.linear_model import SGDClassifier 
    clf = SGDClassifier(loss="hinge", penalty="l1")
    print ("SGD Fitting")
    clf.fit(Xtrain, Ytrain)
    print ("SGD Predicting")
    Ytest = clf.predict(Xtest)
    return Ytest

def write_txt(data, name):
    with open(name,'w',newline = '') as output:
        filewriter = csv.writer(output)
        for result in data:
            filewriter.writerow([result])


if __name__ == "__main__":
    import time
    start = time.time()
    print ("Preprocessing the training_data--")
    imdb_data_preprocess(inpath=train_path, mix=True)
    print ("Done with preprocessing. Now, will retreieve the training data in the required format")
    [Xtrain_text, Ytrain] = grab_train_data()
    print ("Retrieved the training data. Now will retrieve the test data in the required format")
    Xtest_text = grab_test_data(name=test_path)
    print ("Retrieved the test data. Now will initialize the model \n\n")


    print ("-----------------------ANALYSIS ON THE INSAMPLE DATA (TRAINING DATA)---------------------------")
    uni_vectorizer = process_data(Xtrain_text,process_type = 'unigram')
    print ("Fitting the unigram model")
    Xtrain_uni = uni_vectorizer.transform(Xtrain_text)
    print ("After fitting ")
    
    print ("\n")
    print ("-----------------------ANALYSIS ON THE TEST DATA ---------------------------")
    print ("Unigram Model on the Test Data--")
    Xtest_uni = uni_vectorizer.transform(Xtest_text)
    print ("Applying the stochastic descent")
    Ytest_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtest_uni)
    write_txt(Ytest_uni, name="unigram.output.txt")
    print ("Done with  stochastic descent")
    print ("\n")


    #++++++++++++++++++++++++++++++++++++++++++++++
    
    bi_vectorizer = process_data(Xtrain_text,process_type = 'bigram')
    print ("Fitting the bigram model")
    Xtrain_bi = bi_vectorizer.transform(Xtrain_text)
    print ("After fitting ")
    #print ("Applying the stochastic descent")
    #Y_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtrain_bi)
    #print ("Done with  stochastic descent")
    #print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))
    print ("\n")

    uni_tfidf_transformer = process_data_tfidf(Xtrain_uni,process_type = 'unigram')
    print ("Fitting the tfidf for unigram model")
    Xtrain_tf_uni = uni_tfidf_transformer.transform(Xtrain_uni)
    print ("After fitting TFIDF")
    #print ("Applying the stochastic descent")
    #Y_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
    #print ("Done with  stochastic descent")
    #print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
    print ("\n")


    bi_tfidf_transformer = process_data_tfidf(Xtrain_bi,process_type = 'bigram')
    print ("Fitting the tfidf for bigram model")
    Xtrain_tf_bi = bi_tfidf_transformer.transform(Xtrain_bi)
    print ("After fitting TFIDF")
    #print ("Applying the stochastic descent")
    #Y_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
    #print ("Done with  stochastic descent")
    #print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
    print ("\n")


    print ("-----------------------ANALYSIS ON THE TEST DATA ---------------------------")
    print ("Unigram Model on the Test Data--")
    Xtest_uni = uni_vectorizer.transform(Xtest_text)
    print ("Applying the stochastic descent")
    Ytest_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtest_uni)
    write_txt(Ytest_uni, name="unigram.output.txt")
    print ("Done with  stochastic descent")
    print ("\n")


    print ("Bigram Model on the Test Data--")
    Xtest_bi = bi_vectorizer.transform(Xtest_text)
    print ("Applying the stochastic descent")
    Ytest_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtest_bi)
    write_txt(Ytest_bi, name="bigram.output.txt")
    print ("Done with  stochastic descent")
    print ("\n")

    print ("Unigram TF Model on the Test Data--")
    Xtest_tf_uni = uni_tfidf_transformer.transform(Xtest_uni)
    print ("Applying the stochastic descent")
    Ytest_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
    write_txt(Ytest_tf_uni, name="unigramtfidf.output.txt")
    print ("Done with  stochastic descent")
    print ("\n")

    print ("Bigram TF Model on the Test Data--")
    Xtest_tf_bi = bi_tfidf_transformer.transform(Xtest_bi)
    print ("Applying the stochastic descent")
    Ytest_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
    write_txt(Ytest_tf_bi, name="bigramtfidf.output.txt")
    print ("Done with  stochastic descent")
    print ("\n")

    

