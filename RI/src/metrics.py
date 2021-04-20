import numpy as np
import TextRepresenter as TR
from models import ModeleLangue, Okapi
from sklearn.model_selection import KFold
from scipy.stats import t
import matplotlib.pyplot as plt

def score_query(q):
    ps = TR.PorterStemmer()
    return ps.getTextRepresentation(q)

###########################		TME3		################################ 


class EvalMesure():
    def __init__(self):
        pass
    def evalQuery(self,liste,query):
        pass   
    
    
class Rappel(EvalMesure):
    
    def __init__(self,k):
        EvalMesure.__init__(self)
        self.k = k
    def evalQuery(self,liste,query):
        documents_pertinents = query.listPert
        documents_pertinente_rang_k = list(set(liste[:self.k]) & set(documents_pertinents))
        if len(documents_pertinents) == 0:
            return 1
        return len(documents_pertinente_rang_k) / len(documents_pertinents)
    
    
class Precision(EvalMesure):
    def __init__(self,k):
        EvalMesure.__init__(self)
        self.k = k
    def evalQuery(self,liste,query):
        documents_pertinents = query.listPert
        doc_pertinets_au_rang_k = list(set(liste[:self.k]) & set(documents_pertinents))
        return len(doc_pertinets_au_rang_k) / self.k
    
class F_measure(EvalMesure):
    def __init__(self,k,beta):
        EvalMesure.__init__(self)
        self.k = k
        self.beta = beta
    def evalQuery(self,liste,query):
        precision = Precision(self.k).evalQuery(liste,query)
        rappel = Rappel(self.k).evalQuery(liste,query)
        if (precision == 0 and rappel == 0): return 0;
        return (1+self.beta**2)*(precision*rappel) / ((self.beta**2)*(precision + rappel))
    
class PrecisionMoyenne(EvalMesure):
    def __init__(self):
        EvalMesure.__init__(self)
    def evalQuery(self,liste,query):
        if(len(query.listPert) == 0):
            return 1
        presion_moy = 0
        k=1
        for d in liste:
            if d in query.listPert:
                presion_moy +=Precision(k).evalQuery(liste,query)
            k+=1
        return presion_moy/(len(query.listPert))
    
    
    
class ReciprocalRank(EvalMesure): #reciprocal of the rank at which first correct response returned
    def __init__(self):
        EvalMesure.__init__(self)
    def evalQuery(self,liste,query):
        i = 1
        for d in liste :
            if d in query.listPert:
                break
            i += 1
        return 1/i
    
class ndcg(EvalMesure):
    def __init__(self,p):
        EvalMesure.__init__(self)
        self.p = p
    def evalQuery(self,liste,query):
        if len(query.listPert) == 0:
            return 1
        #je suppose que rel est soit 1 soit 0
        dcgp = 0
        if liste[0] in query.listPert:
            dcgp = 1
        for i in range(1,self.p):
            if liste[i] in query.listPert:
                dcgp += 1/np.log2(i+1)
        dcgpi = 1
        for i in range(1,self.p):
            dcgpi += 1/np.log2(i+1)
        return dcgp / dcgpi
    

            
    

class EvalIRModel():

    def __init__(self, collectionQuery, modelIR, k=10, beta=0.5):
        self.k = k
        self.beta = beta
        self.model = modelIR
        self.collectionQuery = collectionQuery
        
    def evalModelAll(self, k=None, beta=None):

        if k is not None:
            self.k = k

        if beta is not None:
            self.beta = beta
            
        evalTypes = [Precision(self.k), Rappel(self.k), F_measure(self.k, self.beta), PrecisionMoyenne(), ReciprocalRank(), ndcg(self.k)]
        
        
        out = [[] for _ in range(len(evalTypes))]

        for query in self.collectionQuery:
            
            liste = [rank[0] for rank in self.model.getRanking(score_query(self.collectionQuery[query].getText()))]

            for i in range(len(evalTypes)):
                out[i].append(evalTypes[i].evalQuery(liste, self.collectionQuery[query]))

        return [(np.mean(l), np.std(l)) for l in out]
    

        
    def evalModel(self, k=None, beta=None, option=3):

        assert(option>=0 and option <6) # option = [0,1,2,3,4,5] as follows [Precision, Rappel, F_measure,
                                                        #                      PrecisionMoyenne, ReciprocalRank, ndcg]
        
        if k is not None:
            self.k = k

        if beta is not None:
            self.beta = beta
            
        evalTypes = [Precision(self.k), Rappel(self.k), F_measure(self.k, self.beta), PrecisionMoyenne(), ReciprocalRank(), ndcg(self.k)]

        l=[]
        for query in self.collectionQuery:
            liste = [rank[0] for rank in self.model.getRanking(score_query(self.collectionQuery[query].getText()))]

            l.append(evalTypes[option].evalQuery(liste, self.collectionQuery[query]))


        return (np.mean(l), np.std(l))
    

    
    
    def sim(self,Model2,alpha,option=3):
        
        m1, std1 = self.evalModel(option=option)
        m2, std2 = EvalIRModel(self.collectionQuery, Model2).evalModel(option=option)


        n = len(self.collectionQuery)
        error1, error2 = std1/np.sqrt(n), std2/np.sqrt(n)

        t_stat = (m1 - m2) / np.sqrt(error1**2.0 + error2**2.0)
        # degrees of freedom
        df = 2*n - 2
    	# calculate the critical value
        cv = t.ppf(1.0 - alpha, df)
        
        if abs(t_stat) <= cv:
        	return True
        return False
    

def optimisationModeleLangue(index, start, end, n, queries, op=3,graph=False):
    
    vals = np.arange(start, end, n)
    lamb_history = []
    for lamb in vals:
        m = ModeleLangue(index,lamb)
        e = EvalIRModel(queries,m)
        lamb_history.append(e.evalModel(option=op)[0]) #prendre la precision moyenne 3

    bestLamb = vals[np.argmax(np.asarray(lamb_history))]
    if graph == True:
        plt.figure()
        plt.plot(vals,lamb_history)
        plt.xlabel("Lamb")
        plt.ylabel("MAP")
        plt.axhline(y=np.max(lamb_history), color='red', linestyle='-')
        plt.axvline(x=bestLamb, color='purple', linestyle='--')
        plt.show()
    return bestLamb


def optimisationOkapi(index, start1, end1, n1, start2, end2, n2, queries, op=3,graph=False):
    
    valsK = np.arange(start1, end1, n1)
    valsK = [round(x, 1) for x in valsK]
    valsB = np.arange(start2, end2, n2)
    valsB = [round(x, 1) for x in valsB]
    
    history = []
    couple_history = []
    for k1 in valsK:
        for b in valsB:
            m = Okapi(index,k1,b)
            e = EvalIRModel(queries,m)
            history.append(e.evalModel(option=op)[0]) #prendre la precision moyenne 3
            couple_history.append((k1,b))
            
    bestCouple = couple_history[np.argmax(np.asarray(history))]
    if graph == True:
        plt.figure(figsize=(20, 10))
        plt.plot(np.arange(len(couple_history)),history)
        plt.xticks(np.arange(len(couple_history)), couple_history, rotation ='vertical')
        plt.xlabel("(k1,b)")
        plt.ylabel("MAP")
        plt.axhline(y=np.max(history), color='red', linestyle='-')
        plt.axvline(x=np.argmax(np.asarray(history)), color='purple', linestyle='--')
        plt.show()
    return bestCouple


def crossValidationModeleLangue(indexer, start, end, n, queries, nb, op=3):
    
    kf = KFold(n_splits=nb,shuffle=True)


    out = []
    
    for train_index, test_index in kf.split(list(queries.keys())):
        
        trainKeys = np.array(list(queries.keys()))[train_index]
        testKeys = np.array(list(queries.keys()))[test_index]
        
        queriesTrain = {k: queries[k] for k in trainKeys}
        queriesTest = {k: queries[k] for k in testKeys}
        
        lamb = optimisationModeleLangue(indexer, start, end, n,queriesTrain,op)
        m = ModeleLangue(indexer,lamb)
        e = EvalIRModel(queriesTest,m)
        out.append(e.evalModel(option=op)[0]) #prendre la precision moyenne 3
        
    return np.array(out)


def crossValidationOkapi(indexer, start1, end1, n1, start2, end2, n2, queries, nb, op=3):
    
    kf = KFold(n_splits=nb,shuffle=True)


    out = []
    
    
    keys = list(queries.keys())
    for train_index, test_index in kf.split(keys):
        
        queriesTrain = {keys[x]:queries[keys[x]] for x in train_index}
        queriesTest = {keys[x]:queries[keys[x]] for x in test_index}

        k1, b = optimisationOkapi(indexer, start1, end1, n1, start2, end2, n2, queriesTrain,op)
        m = Okapi(indexer,k1,b)
        e = EvalIRModel(queriesTest,m)
        out.append(e.evalModel(option=op)[0]) #prendre la precision moyenne 3

    return np.array(out)

