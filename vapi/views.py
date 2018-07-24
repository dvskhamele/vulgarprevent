from django.shortcuts import render

from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from .serializers import UserSerializer, GroupSerializer


import pandas as pd
import numpy as np
import nltk
import string

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import SVC
from sklearn.metrics import classification_report,precision_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import pickle

from rest_framework.views import APIView
from rest_framework.response import Response

from rest_framework.decorators import api_view


def processor(mess):
   
    nopunc = [char for char in mess if char not in string.punctuation]

 
    nopunc = ''.join(nopunc)
    
   
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

@api_view()
def predict(request ):
        stopwords.words('english')[0:10] # Show some stop words
        with open('vapi/data/mymodel.pkl','rb') as f:
            grid = pickle.load(f)
        msg_test = [request.GET["msg"],]
        print("MSG_TEST: ",msg_test)
        predicted = grid.predict(msg_test)
        print(predicted)
        return Response(predicted)

@api_view()
def train(request ):
    data = pd.read_csv('vapi/data/secretanas1.csv', encoding='latin-1', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

    maxval =1 
    data["flags"]=data["flags"].astype(int)
    data["flags"]= data["flags"].where(data["flags"] <= maxval, maxval)

    minval =0 
    data["flags"]= data["flags"].where(data["flags"] >= minval, minval)

    data.drop(data.index[0])

    xs = data.iloc[:, data.columns != 'flags']
    ys = data.iloc[:, data.columns == 'flags']



    number_records_minor = len(data[data.flags == 1])
    minor_indices = np.array(data[data.flags == 1].index)


    normal_indices = data[data.flags == 0].index


    random_normal_indices = np.random.choice(normal_indices, number_records_minor, replace = False)
    random_normal_indices = np.array(random_normal_indices)


    under_sample_indices = np.concatenate([minor_indices,random_normal_indices])


    under_sample_data = data.iloc[under_sample_indices,:]

    #X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'flags']
    #y_undersample = under_sample_data.loc[:, under_sample_data.columns == 'flags']


    mess = 'Sample message! Notice: it has punctuation.'


    nopunc = [char for char in mess if char not in string.punctuation]


    nopunc = ''.join(nopunc)

    stopwords.words('english')[0:10] # Show some stop words

    nopunc.split()

    clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


    pd.options.mode.chained_assignment = None
    under_sample_data["cltxt01"] = under_sample_data["cltxt01"].astype(str)

    under_sample_data["cltxt01"].apply(processor)

    bow_transformer = CountVectorizer(analyzer=processor).fit(under_sample_data["cltxt01"])

    print(len(bow_transformer.vocabulary_))


    text_bow = bow_transformer.transform(under_sample_data["cltxt01"])

    tfidf_transformer = TfidfTransformer().fit(text_bow)

    #tfidf4 = tfidf_transformer.transform(text_bow)

    #flag_detect_model = SVC().fit(tfidf4, under_sample_data["flags"])

    #all_predictions = flag_detect_model.predict(tfidf4)
    #print(all_predictions)

    msg_train, msg_test, label_train, label_test = \
    train_test_split(under_sample_data["cltxt01"], under_sample_data["flags"], test_size=0.2)


    pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=processor)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ support victor classifier
    ])

    parameters = dict(classifier__C=[5,10,20,15],classifier__gamma=[1,0.01,0.1,0.05])

    grid = GridSearchCV(pipeline, verbose=3, param_grid=parameters ,scoring="accuracy")

    grid.fit(msg_train,label_train)

    #grid.best_params_

    #predictionz =grid.predict(msg_test)

    #predictionz = pipeline.predict(msg_test)


    secret = ["i like this thing"]
    with open('vapi/data/mymodel.pkl','wb') as f:
        pickle.dump(grid,f)

    print("DOne")
    predicted = "Done"
    return Response(predicted)
































































class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer