import numpy as np, codecs, json, vocabulary, cPickle as pickle, sys
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support

class lda_gibbs_sampling:
    def __init__(self, K=25, alpha=0.5, beta=0.5, docs= None, V= None):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs # a list of lists, each inner list contains the indexes of the words in a doc, e.g.: [[1,2,3],[2,3,5,8,7],[1, 5, 9, 10 ,2, 5]]
        self.V = V # how many different words in the vocabulary i.e., the number of the features of the corpus
        # Definition of the counters 
        self.z_m_n = [] # topic assignements for each of the N words in the corpus. N: total number od words in the corpus (not the vocabulary size).
        self.n_m_z = np.zeros((len(self.docs), K)) + alpha     # |docs|xK topics: number of words assigned to topic z in document m  
        self.n_z_t = np.zeros((K, V)) + beta # (K topics) x |V| : number of times a word v is assigned to a topic z 
        self.n_z = np.zeros(K) + V * beta    # (K,) : overal number of words assigned to a topic z

        self.N = 0
        for m, doc in enumerate(docs):         # Initialization of the data structures I need.
            self.N += len(doc)                 # to find the size of the corpus 
            z_n = []
            for t in doc: 
                z = np.random.randint(0, K) # Randomly assign a topic to a word. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                z_n.append(z)                  # Keep track of the topic assigned 
                self.n_m_z[m, z] += 1          # increase the number of words assigned to topic z in the m doc.
                self.n_z_t[z, t] += 1          #  .... number of times a word is assigned to this particular topic
                self.n_z[z] += 1               # increase the counter of words assigned to z topic
            self.z_m_n.append(np.array(z_n))# update the array that keeps track of the topic assignements in the words of the corpus.

    def inference(self):
        """    The learning process. Here only one iteration over the data. 
               A loop will be calling this function for as many iterations as needed.     """
        for m, doc in enumerate(self.docs):
            z_n, n_m_z = self.z_m_n[m], self.n_m_z[m]
            for n, t in enumerate(doc):
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1
                # sampling topic new_z for t
                p_z = (self.n_z_t[:, t]+self.beta) * (n_m_z+self.alpha) / (self.n_z + self.V*self.beta)                      # A list of len() = # of topic
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()   # One multinomial draw, for a distribution over topics, with probabilities summing to 1, return the index of the topic selected.
                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1
    def topicdist(self):
        return self.n_m_z / self.n_m_z.sum(axis=1)[:, np.newaxis]
    
    def heldOutPerplexity(self, docs, iterations):
        N, log_per, z_m_n = 0, 0, []
        n_m_z1, n_z_t, n_z = (np.zeros((len(docs), self.K)) + self.alpha), (np.zeros((self.K, self.V)) + self.beta), np.zeros(self.K)
        for m, doc in enumerate(docs):         # Initialization of the data structures I need.
            z_n = []
            try:
                len(doc)
            except:
                print doc
            for t in doc:
                z = np.random.randint(0, self.K) # Randomly assign a topic to a word. Recall, topics have ids 0 ... K-1. randint: returns integers to [0,K[
                z_n.append(z)                  # Keep track of the topic assigned 
                n_m_z1[m, z] += 1          # increase the number of words assigned to topic z in the m doc.
                n_z_t[z, t] += 1          #  .... number of times a word is assigned to this particular topic
                n_z[z] += 1               # increase the counter of words assigned to z topic
            z_m_n.append(np.array(z_n))# update the array that keeps track of the topic assignements in the words of the corpus.
        for i in range(iterations):
            for m, doc in enumerate(docs):
                z_n, n_m_z = z_m_n[m], n_m_z1[m]
                for n, t in enumerate(doc):
                    z = z_n[n]
                    n_m_z[z] -= 1
                    n_z_t[z, t] -= 1
                    n_z[z] -= 1
                    # sampling topic new_z for t
                    p_z = self.n_z_t[:, t] * n_m_z / n_z                       # A list of len() = # of topic
                    new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()   # 1 multinomial draw, return the index of the selected topic
                    # set z the new topic and increment counters
                    z_n[n] = new_z
                    n_m_z[new_z] += 1
                    n_z_t[new_z, t] += 1
                    n_z[new_z] += 1
    
        phi = self.worddist()
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = n_m_z1[m] / (len(doc) + Kalpha)
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], theta))
            N += len(doc)
        topicDist = n_m_z1 / n_m_z1.sum(axis=1)[:, np.newaxis]
        return np.exp(log_per / N), topicDist


    def perplexity(self):
        docs = self.docs
        phi = self.worddist()
        log_per, N = 0, 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(docs[m]) + Kalpha)
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], theta))
            N += len(doc)
        return np.exp(log_per / N)

    def worddist(self):
        """get topic-word distribution, \phi in Blei's paper. Returns the distribution of topics and words. (Z topics) x (V words)  """
        return self.n_z_t / self.n_z[:, np.newaxis]  #Normalize each line (lines are topics), with the number of words assigned to this topics to obtain probs.  *neaxis: Create an array of len = 1


def getClustersKmeans(docs):
    kmeans = KMeans(n_clusters=20, n_init=8, max_iter=300, precompute_distances='auto', verbose=0, copy_x=False, n_jobs=3)
    return kmeans.fit_predict(docs)

def evaluate_clusters(clusters, golden_clusters):
    """
    Find the initial-found cluster mapping.
    Calculate Precision-Recall-F-measure
    """
    assigned_mapping, mapping = {},  {k:[0]*20 for k in range(20)}
    new_clusters = list()
    for key, item in enumerate(clusters):
        try: 
            mapping[item][golden_clusters[key]] += 1
        except:
            print item, key
    for item in range(20):
        #for c in sorted(mapping[item], reverse=True):
        for c in sorted(range(len(mapping[item])), key=lambda k: mapping[item][k], reverse=True):
            if c not in assigned_mapping.values():
                assigned_mapping[item] = c
                break
    for item in clusters:
        new_clusters.append(assigned_mapping[item])
    # Now golden_clusters and k-means clusters speak the same language.
    sco = [0]*20
    recall_deno, prec_deno = [0]*20, [0]*20
    for key, val in enumerate(new_clusters):
        if val == golden_clusters[key]:
            sco[val] +=1
        recall_deno[val] += 1
        prec_deno[golden_clusters[key]] += 1
    sco, recall_deno, prec_deno = np.array(sco, dtype=np.float32), np.array(recall_deno, dtype=np.float32), np.array(prec_deno, dtype=np.float32)
    p, r = np.average(sco/prec_deno), np.average(sco/recall_deno)
    return [p,r, 2*p*r/(p+r)]
    
        
    

if __name__ == "__main__":
    st,corpus = datetime.now(), []
    corpus = codecs.open("toy_dataset.txt", 'r', 'utf-8').read().split('\n')
    print len(corpus), 'Cleaning coprus..'
    for key, val in enumerate(corpus):
        if val == '':
            del(corpus[key])   
    iterations, scores = 500, []
    voca = vocabulary.Vocabulary(excluds_stopwords=False)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    goldenClusters = open('toy_labels.txt').read().splitlines()
    goldenClusters =[ int(x) for x in goldenClusters ]
    #lda = lda_gibbs_sampling(K=int(sys.argv[1]), alpha=0.5, beta=0.5, docs=docs, V=voca.size())
    lda = lda_gibbs_sampling(K=int(sys.argv[1]), alpha=0.01, beta=0.5, docs=docs, V=voca.size())
    for i in range(iterations):
        starting = datetime.now()
        print "iteration:", i, 
        lda.inference()
        print "Took:", datetime.now() - starting
        scores.append(evaluate_clusters(getClustersKmeans(lda.topicdist()), goldenClusters))
    with open('scores.lda.%dtopics.alpha%f.pkl'%int((sys.argv[1]), 0.01), 'w') as out:
        pickle.dump(scores, out )
    """
    d = lda.worddist()
    for i in range(20):
        ind = np.argpartition(d[i], -15)[-15:] # an array with the indexes of the 10 words with the highest probabilitity in the topic
        for j in ind:
            print voca[j],
        print 
    """
    print "It finished. Total time:", datetime.now()-st
        
        
        
