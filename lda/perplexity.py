import numpy as np, vocabulary, string, nltk.data, sys, codecs, json, time
from nltk.tokenize import sent_tokenize
from lda import lda_gibbs_sampling
from datetime import datetime
from sklearn.cross_validation import train_test_split, StratifiedKFold
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.utils import shuffle
from functions import *

path2training = sys.argv[1]
training = codecs.open(path2training, 'r', encoding='utf8').read().splitlines()

topics = int(sys.argv[2])
alpha, beta = 0.5 / float(topics), 0.5 / float(topics)

voca_en = vocabulary.Vocabulary(set(nltk.corpus.stopwords.words('english')), WordNetLemmatizer(), excluds_stopwords=True)

ldaTrainingData = change_raw_2_lda_input(training, voca_en, True)
ldaTrainingData = voca_en.cut_low_freq(ldaTrainingData, 1)

classificationData, y = load_classification_data(sys.argv[3], sys.argv[4])
classificationData = change_raw_2_lda_input(classificationData, voca_en, False)
classificationData = voca_en.cut_low_freq(classificationData, 1)

iterations = 201
start = time.time()
final_acc, final_mif, final_perpl, final_ar, final_nmi, final_p, final_r, final_f = [], [], [], [], [], [], [], []
for j in range(5):
    perpl, cnt, acc, mif, ar, nmi, p, r, f = [], 0, [], [], [], [], [], [], []
    lda = lda_gibbs_sampling(K=topics, alpha=alpha, beta=beta, docs= ldaTrainingData, V=voca_en.size())
    for i in range(iterations):
        lda.inference()
        if i % 5 == 0:
            print "Iteration:", i, "Perplexity:", lda.perplexity(),
            features = lda.heldOutPerplexity(classificationData, 3)
            print "Held-out:", features[0]
            scores = perform_class(features[1], y)
            acc.append(scores[0][0])
            mif.append(scores[1][0])
            perpl.append(features[0])
    final_acc.append(acc)
    final_mif.append(mif)
    final_perpl.append(perpl)

np.save("results/lda.acc.%dtopics.%s"%(topics, sys.argv[5]), np.array(final_acc, dtype=np.float) )
np.save("results/lda.mif.%dtopics.%s"%(topics, sys.argv[5]), np.array(final_mif, dtype=np.float) )
np.save("results/lda.perpl.%dtopics.%s"%(topics, sys.argv[5]), np.array(final_perpl, dtype=np.float))

elapsed = time.time() - start
print "Timing:", elapsed
