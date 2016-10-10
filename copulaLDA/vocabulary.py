import nltk, re, string
from nltk.corpus import conll2000
from nltk.tokenize import sent_tokenize
import numpy as np
recover_list = {"wa":"was", "ha":"has"}


class VocabularySentenceLayer:
    def __init__(self, stopwords, wl, excluds_stopwords=False):
        self.vocas = []        # id to word
        self.vocas_id = dict() # word to id
        self.docfreq = []      # id to document frequency
        self.excluds_stopwords = excluds_stopwords
        self.stopwords = stopwords
        self.wl =  wl
        self.table = {ord(c): None for c in string.punctuation}

    def is_stopword(self, w):
        return w in self.stopwords

    def lemmatize(self, w0):
        w0 = w0.translate(self.table)
        w = self.wl.stem(w0.lower())
        if w in recover_list:
            return recover_list[w]
        return w

    def term_to_id(self, term0, training):
        term = self.lemmatize(term0)
        if not re.match(r'[a-z]+$', term): 
            return None
        if self.excluds_stopwords and self.is_stopword(term): 
            return None
        try:  
            term_id = self.vocas_id[term]
        except:
            if not training: 
                return None
            term_id = len(self.vocas)
            self.vocas_id[term] = term_id
            self.vocas.append(term)
            self.docfreq.append(0)
        return term_id

    def doc_to_ids(self, doc, training=True):
        l = []
        words = dict()
        window = 150
#        doc = doc.replace("&ndash;", " ")
#        doc = sent_tokenize(doc)
        for sentence in doc:
            miniArray = []
            for term in sentence:
                id = self.term_to_id(term, training)    
                if id != None:
                    miniArray.append(id)
                    if not id in words:
                        words[id] = 1
                        self.docfreq[id] += 1
            if not len(miniArray):
                continue
            if len(miniArray)  > window:
                l.extend([np.array(miniArray[i:i+window]) for i in xrange(0, len(miniArray), window)])
            else:
                l.append(np.array(miniArray))
        return l


        
    def cut_low_freq(self, corpus, threshold=1):
        new_vocas = []
        new_docfreq = []
        self.vocas_id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocas):
            freq = self.docfreq[id]
            if freq > threshold:
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocas = new_vocas
        self.docfreq = new_docfreq
        return np.array([ self.conv(doc, conv_map) for doc in corpus])

    def conv(self, doc, conv_map, window=10):
        n = [np.array([conv_map[id] for id in sen if id in conv_map]) for sen in doc]
        n = [x for x in n if x.shape[0] > 0]
        m = []
        for x in n:
            if x.shape[0] > window:
                m.extend([x[i:i+window] for i in xrange(0, x.shape[0], window)])
            else:
                m.append(x)
        return  np.array(m)    

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in xrange(0, l.shape[0], n):
            yield l[i:i+n]



    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)

    def is_stopword_id(self, id):
        return self.vocas[id] in stopwords_list


class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): # [_code-unigram-chunker-constructor]
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data) # [_code-unigram-chunker-buildit]

    def parse(self, sentence): # [_code-unigram-chunker-parse]
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


class ConsecutiveNPChunkTagger(nltk.TaggerI): # [_consec-chunk-tagger]

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) # [_consec-use-fe]
                train_set.append( (featureset, tag) )
                history.append(tag)
#        self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam', trace=0)
        self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): # [_consec-chunker]
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

def tags_since_dt(sentence, i):
     tags = set()
     for word, pos in sentence[:i]:
         if pos == 'DT':
             tags = set()
         else:
             tags.add(pos)
     return '+'.join(sorted(tags))


def npchunk_features(sentence, i, history):
     word, pos = sentence[i]
     if i == 0:
         prevword, prevpos = "<START>", "<START>"
     else:
         prevword, prevpos = sentence[i-1]
     if i == len(sentence)-1:
         nextword, nextpos = "<END>", "<END>"
     else:
         nextword, nextpos = sentence[i+1]
     return {"pos": pos,
             "word": word,
             "prevpos": prevpos,
             "nextpos": nextpos, 
             "prevpos+pos": "%s+%s" % (prevpos, pos), 
             "pos+nextpos": "%s+%s" % (pos, nextpos),
             "tags-since-dt": tags_since_dt(sentence, i)} 
