from test import tf_idf
from test import feature_values
from os import listdir
from sklearn import svm
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import math

cachedStopWords = stopwords.words("english")
min_lenght = 4

class corpus:
    def __init__(self, dir_pos, dir_neg):
        self.dir_pos = dir_pos
        self.dir_neg = dir_neg
        self.documents = []
        self.vocabulary = {}
        self.inversed_vocabulary = {}

        for i, file in enumerate(listdir(dir_pos)):

            if i < 300:
                fs = open(dir_pos + "\\" + file, 'r')
                text = fs.read()
                positive = 1
                train = 0
                self.documents.append(document(text, positive, train))

            else:
                fs = open(dir_pos + "\\" + file, 'r')
                text = fs.read()
                positive = 1
                train = 1
                self.documents.append(document(text, positive, train))
                # posdocs.append(open(dir_pos + "\\" + file, 'r').read())

        for i, file in enumerate(listdir(dir_neg)):

            fs = open(dir_neg + "\\" + file, 'r')
            text = fs.read()

            if i < 300:
                self.documents.append(document(text, 0, 0))
            else:
                self.documents.append(document(text, 0, 1))

        for i, doc in self.documents:
            for word in doc.get_unique_words():
                if not word in self.vocabulary:
                    self.vocabulary[i] = word
                    self.inversed_vocabulary[i] = 1

    def add_document(self, document):
        self.documents.append(document)

    def get_train_documents(self):
        train = []
        for doc in self.documents:
            if doc.train == 1:
                train.append(doc.text)
        return train

    def get_representer(self):
        return tf_idf(self.get_train_documents())

    #def get_vocabulary(self):

    def get_svm_vectors(self, train = 0, test = 0):

        xs = []
        ys = []

        for doc in self.documents:
            if train == 1:
               if doc.train == 1:
                   continue

            if test == 1:
               if doc.text == 1:
                   continue

            x = doc.get_vector(self.inversed_vocabulary)
            y = doc.positive

            xs.append(x)
            ys.append(y)

        return (xs, ys)

class document:
    def __init__(self, text, positive = 1, train = 1):
        self.positive = positive
        self.train = train
        self.text = text

    def get_features_values(self, representer):
        return feature_values(self.text, representer)

    def preProcessing(self, routers):
        stemmed_tockens = []
        stemmer = PorterStemmer()

        raw_tocken = routers.split()
        no_stopwords = [token for token in raw_tocken if token not in cachedStopWords]

        for token in no_stopwords:
            stemmed_tockens.append(stemmer.stem(token and len(token)>= min_lenght))

        p = re.compile('[a-zA-Z]+');

        filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_lenght, stemmed_tockens));

        return filtered_tokens

    def get_unique_words(self):
        words = []

        for word in preProcessing(self.text.split()) :
            if not word in words:
                words.append(word)
        return  words

    def get_vector(self, inverse_vocabulary):
        lng = len(inverse_vocabulary)
        vector = [0 for i in range(lng)]
        for word in self.text.split():
           vector[inverse_vocabulary[word]] = 1

class tf-idf :
    def __init__(self):
        self.D = 0
        self.df = {}
        self.idf_vocabulary = {}

    def add_doc(self, tokens):
        for token in set(tokens):
            self.D += 1
            self.df(token) += 1

    def idf(self, token):
        return math.log(self.D/self.df[token])

    def tf(self,token, docs):
        l_wyst_tokenu = 0.0
        l_tokenow = 0.0

        for t in docs:
            l_tokenow += 1.0
            if t == token:
                l_wyst_tokenu += 1.0

        return l_wyst_tokenu/ l_tokenow

    def tfidf(self, token, dock):
        return  tf(token, dock) * idf(token)




klasyfikator = svm.SVC(kernel= "linear")
crp = corpus("C:\\Users\\s0152857\\Downloads\\txt_sentoken\\neg", "C:\\Users\\s0152857\\Downloads\\txt_sentoken\\pos")
(X, y) = crp.get_svm_vectors(train = 1)
klasyfikator.fit(X,y)

pozytywne = 0
wszystkie = 0

for i,x in enumerate(X):
    wszystkie += 1
    klasa = klasyfikator.predict(x)
    if klasa == y[i]:
        pozytywne = pozytywne + 1

print(pozytywne)
print(wszystkie)

#print(corpus.documents[0].get_features_values())
