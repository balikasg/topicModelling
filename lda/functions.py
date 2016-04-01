import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from nltk.stem.snowball import SnowballStemmer
import numpy as np, string, nltk.data, codecs
from nltk.tokenize import sent_tokenize
from datetime import datetime
from sklearn import svm, cluster, metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
from collections import Counter, defaultdict
from sklearn import preprocessing


def change_raw_2_lda_input(raw_data, voca_en, training):
    data2return = []
    for d in raw_data:
        data2return.append(np.array(voca_en.doc_to_ids(d,training=training), dtype=object))
    assert len(data2return)==len(raw_data)
    return np.array(data2return)

def load_classification_data(path2train, path2labels):
    X = codecs.open(path2train, 'r', encoding='utf8').read().splitlines()
    y = codecs.open(path2labels, 'r', encoding='utf8').read().splitlines()
    X,y =shuffle(X,y, random_state=123)
    classes = Counter(y)
    XX,yy = [], []
    for k, v in enumerate(y):
        if classes[v]>5:
            XX.append(X[k])
            yy.append(v)
    print "Classification problem:\n", len(set(yy)), 'classes.', len(XX), "instances."
    return XX, np.array(yy).astype(int)



def perform_class(X, y, iterations=1):
    scores = []
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42+iterations)
        parameters = {'C':[0.01, 0.1, 1, 10, 100]}
        clf_acc = GridSearchCV(svm.LinearSVC(), parameters, n_jobs=3, cv=3, refit=True, scoring = 'accuracy')
        clf_acc.fit(X_train, y_train)
        scores.append([metrics.accuracy_score(y_test, clf_acc.predict(X_test)), metrics.f1_score(y_test, clf_acc.predict(X_test),average='micro')])
    acc = np.mean([x[0] for x in scores]), np.std([x[0] for x in scores])
    mif = np.mean([x[1] for x in scores]), np.std([x[1] for x in scores])
    return acc, mif


