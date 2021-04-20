import numpy as np

#######################################  TME2  ################################################

class Weighter():
    
    def __init__(self,index):
        self.index=index   
        
    def getWeightsForDoc(self,idDoc):
        return self.index.getTfsForDoc(idDoc)
    
    def getWeightsForStem(self,stem):
        return self.index.getTfsForStem(stem)
    
    def getWeightsForQuery(self,query):
        pass 
    
    
class Weighter1(Weighter):
    
    def __init__(self, index):
        Weighter.__init__(self, index)
        
    def getWeightsForQuery(self,query):
        out = dict()
        for terme in query.keys():
            out[terme] = 1
        return out
    
    
class Weighter2(Weighter):
    
    def __init__(self, index):
        Weighter.__init__(self, index)
        
    def getWeightsForQuery(self,query):
        return query
    
    
class Weighter3(Weighter):
    
    def __init__(self, index):
        Weighter.__init__(self, index)
        
    def getWeightsForQuery(self,query):
        out = dict()
        for terme in query.keys():
            out[terme] = self.index.getIDFsForStem(terme)
        return out
    
    
class Weighter4(Weighter):
    
    def __init__(self, index):
        Weighter.__init__(self, index)
        
    def getWeightsForDoc(self,idDoc):
        tfs_doc = self.index.getTfsForDoc(idDoc)
        return {t:(1 + np.log(tfs_doc[t])) for t in tfs_doc.keys()}
    
    def getWeightsForStem(self,stem):
        tfs_stem = self.index.getTfsForStem(stem)
        return {t:(1 + np.log(tfs_stem[t])) for t in tfs_stem.keys()}
    
    def getWeightsForQuery(self,query):
        out = dict()
        for term in query.keys():
            out[term] = self.index.getIDFsForStem(term)
        return out
    
    
class Weighter5(Weighter):
    
    def __init__(self, index):
        Weighter.__init__(self, index)
        
    def getWeightsForDoc(self,idDoc):
        tfs_doc = self.index.getTfsForDoc(idDoc)
        idf_stem = self.index.getIDFsForStem(idDoc)
        return {t:(1 + np.log(tfs_doc[t]))*idf_stem for t in tfs_doc.keys()}
    
    def getWeightsForStem(self,stem):
        tfs_doc = self.index.getTfsForStem(stem)
        idf_stem = self.index.getIDFsForStem(stem)
        if stem in self.index.getIndexInv().keys():
            return {t:(1 + np.log(tfs_doc[t]))*idf_stem for t in tfs_doc.keys()}
        else:
            return {}
    
    def getWeightsForQuery(self,query):
        out = dict()
        for terme in query.keys():
            idf_stem = self.index.getIDFsForStem(terme)
            tfs_doc = query[terme]
            out[terme] = (1+np.log(tfs_doc))*idf_stem
        return out
    

class IRModel():
    
    def __init__(self, index):
        self.index=index 
        
    def getScores(self,query):
        pass 
    
    def getRanking(self,query):
        score_docs = self.getScores(query)        
        score_docs = {id:score for id,score in score_docs.items() if score != 0}
        sore_docs_sorted = sorted(score_docs.items(),reverse = True, key=lambda x: x[1])
        return sore_docs_sorted
    
	

class Vectoriel(IRModel):
    
    def __init__(self, index,Weighter,normalized):
        
        IRModel.__init__(self, index) 
        self.Weighter=Weighter
        self.normalized=normalized
        self.doc_weights = {idDoc : self.Weighter.getWeightsForDoc(idDoc) for idDoc in self.index.getIndex().keys()}
        self.doc_norm = {idDoc : np.linalg.norm(np.array(list(doc.values()))) for (idDoc, doc) in self.doc_weights.items()}
    

    
    def getScores(self,query):
        out=dict()
        wtq = self.Weighter.getWeightsForQuery(query)
        normQ = np.linalg.norm(np.array(list(wtq.values())))
        for (idDoc, doc_w) in self.doc_weights.items():
            out[idDoc] = 0
            for (t, t_w) in wtq.items():
                if t in doc_w:
                    out[idDoc] = out.get(idDoc,0) + (doc_w[t] * t_w)
                if self.normalized:
                    out[idDoc] = out.get(idDoc,0) / (normQ + self.doc_norm[idDoc])
              
        return out  

            
  
class ModeleLangue(IRModel):
    
    def __init__(self, index, lamb=.8):
        super().__init__(index)
        self.lamb = lamb
        self.collectionSize = self.index.getCollectionSize()
		
    def getScores(self, query):

        scores = dict()
        
        for t,nb in query.items():
            if t in self.index.getIndexInv().keys():
                tf = self.index.getTfsForStem(t)
                pcoll = sum(tf.values())/self.collectionSize

                
                for idDoc,doc in self.index.getIndex().items():
                    if t in doc.keys():
                        pdoc = tf[idDoc] / self.index.getDocumentSize(idDoc) 
                    else:
                        pdoc = 0
                    
                    s = (1-self.lamb)*pdoc+self.lamb*pcoll
                    if s>0:
                        scores[idDoc] = scores.get(idDoc,0)+np.log(s)
                    
        return scores
        


class Okapi(IRModel):
    
    def __init__(self, index, k1=1.2, b=0.75):
        IRModel.__init__(self, index)
        self.k1 = k1
        self.b = b
        self.moy = self.index.getCollectionSize() / len(self.index.getIndex())
		
    def getScores(self, query, pertinences=None):

		
        scores = dict()
        for idDoc in self.index.getIndex().keys() :
            score = 0
            tf_doc = self.index.getTfsForDoc(idDoc)
            for t in query.keys():
                if (t in tf_doc):
                    idf = self.index.getIDFsForStem(t)
                    tf = tf_doc[t]
                    score += idf * (tf / (tf + self.k1 * (1 - self.b + self.b * ( self.index.getDocumentSize(idDoc) / self.moy))))
            scores[idDoc]=score
        return scores