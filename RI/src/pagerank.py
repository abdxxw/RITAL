

import numpy as np

import matplotlib.pyplot as plt
            

def sousGraphe(model,query,k,n):

    VQ=set(np.array(model.getRanking(query))[:,0][:n])
    for doc in VQ : 
        VQ=VQ.union(model.index.getHyperlinksFrom(doc))
        VQ=VQ.union(np.random.choice(list(model.index.getHyperlinksTo(doc).keys()), k))
    VQ = list(map(str,VQ))
    return VQ

def pagerank(model,query,d,k,n,eps,nbiter):
    VQ=sousGraphe(model,query,k,n)
    graph=dict()
    tab_score=dict()
    tab_score_new=dict()
    aj=1
    for i in VQ : 
        graph[i]=list(set(model.index.getHyperlinksFrom(i)) & set(VQ))
        tab_score[i]=1/len(VQ)
    for j in range(nbiter):      
        for i in VQ:
            s=0
            for j in graph.keys():
                if i in graph[j]:
                    s=s+tab_score[j]/len(graph[j])
            tab_score_new[i]=d*s+(1-d)*aj
        somme_proba = np.sum(list(tab_score_new.values()))
        for i in VQ:
            tab_score_new[i] /= somme_proba
        if np.sum(np.abs(np.array(list(tab_score_new.values()))-np.array(list(tab_score.values())))) < eps :
            break
        tab_score=tab_score_new
    tr = np.array(list(tab_score_new.values()), dtype = np.float32)
    sort = np.flip(np.argsort(tr))
    return np.array(list(tab_score_new.keys()))[sort] , np.array(list(tab_score_new.values())).mean()
        
  
def optimisationD(model,start1,end1,n1,query,k,n,eps,maxiter,graph=False):
    
    valsd = np.arange(start1, end1, n1) 
    history = []
    for d in valsd:       
        docs,score = pagerank(model,query,d,k,n,eps,maxiter)
        history.append(score)
    best_d = valsd[np.argmax(np.asarray(history))]
    if graph == True:
        plt.figure()
        plt.plot(valsd,history)
        plt.xlabel("d value")
        plt.ylabel("score")
        plt.axhline(y=np.max(history), color='red', linestyle='-')
        plt.axvline(x=best_d, color='purple', linestyle='--')
        plt.show()
    return best_d
