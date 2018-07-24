
from nltk.corpus import stopwords



def processor(mess):
   
    nopunc = [char for char in mess if char not in string.punctuation]

 
    nopunc = ''.join(nopunc)
    
   
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


import pickle
with open('vapi/data/mymodel.pkl','rb') as f:
    stopwords.words('english')[0:10] # Show some stop words
    grid = pickle.load(f)


