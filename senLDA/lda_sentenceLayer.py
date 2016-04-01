import numpy as np, codecs, json,  cPickle as pickle, sys
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import warnings
warnings.filterwarnings("error")

class lda_gibbs_sampling1:
    def __init__(self, K=25, alpha=0.5, beta=0.5, docs= None, V= None):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs # a list of lists, each inner list contains the indexes of the words in a doc, e.g.: [[1,2,3],[2,3,5,8,7],[1, 5, 9, 10 ,2, 5]]
        self.V = V # how many different words in the vocabulary i.e., the number of the features of the corpus
        # Definition of the counters 
        self.z_m_n = [] # topic assignements for each of the N words in the corpus. N: total number of words in the corpus (not the vocabulary size).
        self.n_m_z = np.zeros((len(self.docs), K), dtype=np.float64) + alpha     # |docs|xK topics: number of sentences assigned to topic z in document m  
        self.n_z_t = np.zeros((K, V), dtype=np.float64) + beta # (K topics) x |V| : number of times a word v is assigned to a topic z 
        self.n_z = np.zeros(K) + V * beta    # (K,) : overal number of words assigned to a topic z
        self.N = 0
        for m, doc in enumerate(docs):         # Initialization of the data structures I need.
            z_n = []
            for sentence in doc: 
                self.N += len(sentence)
                z = np.random.randint(0, K) # Randomly assign a topic to a sentence. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                z_n.append(z)                  # Keep track of the topic assigned 
                self.n_m_z[m, z] += 1          # increase the number of sentences assigned to topic z in the m doc.
                self.n_z_t[z, sentence.astype(dtype=np.int32)] += 1   #  .... number of times a word is assigned to this particular topic
                self.n_z[z] += len(sentence)   # increase the counter of words assigned to z topic
            self.z_m_n.append(np.array(z_n)) # update the array that keeps track of the topic assignements in the sentences of the corpus.

    def heldOutPerplexity(self, docs, iterations):
        N, log_per, z_m_n = 0, 0, []
        n_m_z1, n_z_t, n_z = (np.zeros((len(docs), self.K)) + self.alpha), (np.zeros((self.K, self.V)) + self.beta), np.zeros(self.K)
        for m, doc in enumerate(docs):         # Initialization of the data structures I need.
            z_n = []
            for sentence in doc:
                z = np.random.randint(0, self.K) # Randomly assign a topic to a sentence. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                z_n.append(z)                  # Keep track of the topic assigned 
                n_m_z1[m, z] += 1          # increase the number of sentences assigned to topic z in the m doc.
                n_z_t[z, sentence.astype(dtype=np.int32)] += 1   #  .... number of times a word is assigned to this particular topic
                n_z[z] += len(sentence)   # increase the counter of words assigned to z topic
            z_m_n.append(np.array(z_n)) # update the array that keeps track of the topic assignements in the sentences of the corpus.
        for i in range(iterations):
            for m, doc in enumerate(docs):
                z_n, n_m_z = z_m_n[m], n_m_z1[m]
                for sid, sentence in enumerate(doc):
                    z = z_n[sid] # Obtain the topic that was assigned to sentences
                    n_m_z[z] -= 1 # Decrease the number of the sentences in the current document assigned to topic z
                    n_z_t[z, sentence.astype(dtype=np.int32)] -= 1 #Decrease the number of the words assigned to topic z
                    n_z[z] -= len(sentence)  # Decrease the total number of words assigned to topic z
                    p_z = self.get_full_conditional(sentence, m, z, n_z, n_m_z1)
                    new_z = np.random.multinomial(1, p_z).argmax()
                    z_n[sid] = new_z
                    n_m_z[new_z] += 1
                    n_z_t[new_z, sentence.astype(dtype=np.int32)] += 1
                    n_z[new_z] += len(sentence)
        phi = self.worddist()
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = n_m_z1[m] / (len(doc) + Kalpha)
            for sen in doc:
                for w in sen:
                    log_per -= np.log(np.inner(phi[:,w], theta))
                N += len(sen)
        topicDist = n_m_z1 / n_m_z1.sum(axis=1)[:, np.newaxis]
        return np.exp(log_per / N), topicDist

    def get_full_conditional(self, sentence, m, z, n_z, n_m_z):
        prod_nom, prod_den = [] , []
        words = Counter(sentence)
        for key, val in words.iteritems():
            for x in range(val):
                quantity = self.n_z_t[:,key] + self.beta + x
                prod_nom.append(quantity)
#                prod_nom *= (quantity)
        prod_nom = np.array(prod_nom, dtype=np.float128)
        left_denominator = n_z + self.beta*self.V
        for x in range(len(sentence)):
            quantity = left_denominator + x
            prod_den.append(quantity)
#            prod_den *= (quantity)
        prod_den = np.array(prod_den, dtype=np.float128)
#        print "Shapes of interest:", prod_den.shape, prod_nom.shape
        prodall1 = np.divide(prod_nom,prod_den)
#        print "After division:", prodall.shape
        prodall = np.prod(prodall1, axis=0)
#        print "After multiplication", prodall.shape
#        prod_nom = np.prod(prod_nom, axis=0, dtype=np.float128)
#        prod_den = np.prod(prod_den, axis=0, dtype=np.float128)

#        left = prod_nom/prod_den
        right = (n_m_z[m,:] + self.alpha)
        p_z = prodall*right
#        try:    
#            p_z /= np.sum(p_z)
#        except:
#            print p_z
#            print prodall1
#            print prodall
        p_z /= np.sum(p_z)
#        except RuntimeWarning:
#            print 'Exception'
#            print prodall 
#            print right
#            print self.n_z_t[:,key]
        return p_z.astype(np.float64)




    def inference(self):
        """    The learning process. Here only one iteration over the data. 
               A loop will be calling this function for as many iterations as needed.     """
        for m, doc in enumerate(self.docs):
            z_n, n_m_z = self.z_m_n[m], self.n_m_z[m] #Take the topics of the sentences and the number of sentences assigned to each topic
            for sid, sentence in enumerate(doc):
                z = z_n[sid] # Obtain the topic that was assigned to sentences
                n_m_z[z] -= 1 # Decrease the number of the sentences in the current document assigned to topic z
                self.n_z_t[z, sentence.astype(dtype=np.int32)] -= 1 #Decrease the number of the words assigned to topic z
                self.n_z[z] -= len(sentence)  # Decrease the total number of words assigned to topic z
                # Get full conditional to sample from
                p_z = self.get_full_conditional(sentence, m, z, self.n_z, self.n_m_z)
                new_z = np.random.multinomial(1, p_z ).argmax()
                z_n[sid] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, sentence.astype(dtype=np.int32)] += 1
                self.n_z[new_z] += len(sentence)
    

    def topicdist(self):
        return self.n_m_z / self.n_m_z.sum(axis=1)[:, np.newaxis]


    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(doc) + Kalpha)
            for sen in doc:
                for w in sen:
                    log_per -= np.log(np.inner(phi[:,w], theta))
                N += len(sen)
        return np.exp(log_per / N)

    def worddist(self):
        """get topic-word distribution, \phi in Blei's paper. Returns the distribution of topics and words. (Z topics) x (V words)  """
        return self.n_z_t / self.n_z[:, np.newaxis]  #Normalize each line (lines are topics), with the number of words assigned to this topics to obtain probs.  *neaxis: Create an array of len = 1


