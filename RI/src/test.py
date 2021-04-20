from collection import Parser, IndexerSimple, QueryParser
from models import Weighter1, Weighter2, Weighter3, Weighter4, Weighter5, Vectoriel, ModeleLangue, Okapi
from metrics import optimisationModeleLangue, optimisationOkapi, EvalIRModel,crossValidationModeleLangue,crossValidationOkapi
from pagerank import pagerank, optimisationD
import TextRepresenter as TR
import random
import itertools 





# =================== parser and index =================================

print("\n#################### PARSING ###################\n")
parser = Parser()
docs = parser.parse("..\data\cisi\cisi.txt")

qParser = QueryParser()
queries = qParser.parse("..\data\cisi\cisi.qry","..\data\cisi\cisi.rel")

print("Done...")

print("\n#################### Indexing ###################\n")
indexer =  IndexerSimple()
indexer.indexation(docs)

print("Done...")



def score_query(q):
    ps = TR.PorterStemmer()
    return ps.getTextRepresentation(q)

def test_model(model,query):
    query = score_query(query)
    hits = model.getRanking(query)
    print("\nMost Relevent Document :\n")
    print(docs[hits[0][0]].getText())

def eval_model(model,queries):
    e = EvalIRModel(queries,model)
    print("Evaluation :")
    metrics = [x[0] for x in e.evalModelAll()]
    
    print("[Precision = {}, Rappel = {}, F_measure = {}, PrecisionMoyenne = {}, ReciprocalRank = {}, ndcg = {}]" \
          .format(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5]))

        
def split(myDict, p):
    
    l = list(myDict.items())
    random.shuffle(l)
    myDict = dict(l)

    n = int(len(myDict) * p)    
    i = iter(myDict.items())     

    train = dict(itertools.islice(i, n))   
    test = dict(i)                        

    return train, test

print("\n#################### Testing Model hits ####################\n")  
  
q = queries[list(queries.keys())[31]].getText()
print("\nQuery : ", q)
'''
print("\n===================================\n\nTest of Vectoriel model with Weighter1 : ")
weighter = Weighter1(indexer)
model = Vectoriel(indexer,weighter,False)

test_model(model,q)

eval_model(model,queries)

print("\n===================================\n\nTest of Vectoriel model with Weighter2 : ")
weighter = Weighter2(indexer)
model = Vectoriel(indexer,weighter,False)

test_model(model,q)

eval_model(model,queries)

print("\n===================================\n\nTest of Vectoriel model with Weighter3 : ")
weighter = Weighter3(indexer)
model = Vectoriel(indexer,weighter,False)

test_model(model,q)

eval_model(model,queries)

print("\n===================================\n\nTest of Vectoriel model with Weighter4 : ")
weighter = Weighter4(indexer)
model = Vectoriel(indexer,weighter,False)

test_model(model,q)

eval_model(model,queries)


print("\n===================================\n\nTest of Vectoriel model with Weighter5 : ")
weighter = Weighter5(indexer)
model = Vectoriel(indexer,weighter,False)

test_model(model,q)

eval_model(model,queries)

print("\n===================================\n\nTest of Language model with lamb = 0.7 : ")

model = ModeleLangue(indexer,0.7)

test_model(model,q)

eval_model(model,queries)
print("\n===================================\n\nTest of Okapi BM25 model with k1 = 1.4 and b= 0.8: ")

model = Okapi(indexer,1.4,0.8)

test_model(model,q)

eval_model(model,queries)




print("\n#################### GridSearch ####################\n")  

trainDict , testDict = split(queries,0.8)


startlamb, endlamb, steplamb = 0.1, 1, 0.1

startk, endk, stepk = 1, 2, 0.1

startb, sendb, stepb = 0.1, 1, 0.1

print("\n===================================\n\n ModeleLangue Searching for best lamb... ")

lamb = optimisationModeleLangue(indexer,startlamb,endlamb,steplamb,trainDict,graph=True)

print("Best lamb : "+str(lamb))

m = ModeleLangue(indexer,lamb)

e = EvalIRModel(testDict,m)
print("Evaluation :")
metrics = [x[0] for x in e.evalModelAll()]

print("[Precision = {}, Rappel = {}, F_measure = {}, PrecisionMoyenne = {}, ReciprocalRank = {}, ndcg = {}]" \
      .format(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5]))


print("\n===================================\n\n BM25 Searching for best k1 and b... ")

k1 , b = optimisationOkapi(indexer,startk, endk, stepk, startb, sendb, stepb,trainDict,graph=True)

print("Best params k1, b : "+str(k1)+", "+str(b))

m = Okapi(indexer,k1,b)

e = EvalIRModel(testDict,m)
print("Evaluation :")
metrics = [x[0] for x in e.evalModelAll()]

print("[Precision = {}, Rappel = {}, F_measure = {}, PrecisionMoyenne = {}, ReciprocalRank = {}, ndcg = {}]" \
      .format(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5]))

  

print("\n===================================\n\n ModeleLangue Cross Validation ... ")
  

values = crossValidationModeleLangue(indexer,startlamb, endlamb, steplamb, queries, 5)
print("MAP of each fold : ")
print(values)
print("Mean MAP ModeleLangue: "+str(values.mean()))

print("\n===================================\n\n BM25 Cross Validation ... ")

values = crossValidationOkapi(indexer,startk, endk, stepk, startb, sendb, stepb, queries, 5)
print("MAP of each fold : ")
print(values)
print("Mean MAP BM25: "+str(values.mean()))


print("\n===================================\n\n models similarity ... ")

    

import numpy as np

import matplotlib.pyplot as plt
model_names = ["Vectoriel1", "Vectoriel2", "Vectoriel3", "Vectoriel4", "Vectoriel5", "ModeleLangue", "Okapi BM25"]

models = [Vectoriel(indexer,Weighter1(indexer),True), \
          Vectoriel(indexer,Weighter2(indexer),True), \
          Vectoriel(indexer,Weighter3(indexer),True), \
          Vectoriel(indexer,Weighter4(indexer),True), \
          Vectoriel(indexer,Weighter5(indexer),True), \
          ModeleLangue(indexer,0.9), \
          Okapi(indexer,1.7,0.8)]
    
similarity_matrix = []
alpha = 0.4
for m in models:
    e = EvalIRModel(queries,m)
    similar = []
    for m2 in models:
        b = e.sim(m2,alpha)
        similar.append(b)
    similarity_matrix.append(similar)


fig, ax = plt.subplots()
im = ax.imshow(similarity_matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(len(model_names)))
ax.set_yticks(np.arange(len(model_names)))
# ... and label them with the respective list entries
ax.set_xticklabels(model_names)
ax.set_yticklabels(model_names)



    
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


plt.show()

print("\n===================================\n\n PageRank ... ")
'''

start1, end1, n1 = 0.1, 1, 0.1
k=2
n=2
eps = 0.1
maxiter = 1000

models = [Vectoriel(indexer,Weighter1(indexer),True), \
          Vectoriel(indexer,Weighter2(indexer),True), \
          Vectoriel(indexer,Weighter3(indexer),True), \
          Vectoriel(indexer,Weighter4(indexer),True), \
          Vectoriel(indexer,Weighter5(indexer),True), \
          ModeleLangue(indexer,0.9), \
          Okapi(indexer,1.7,0.8)]
    


qq = score_query(q)

d = optimisationD(models[6],start1,end1,n1,qq,k,n,eps,maxiter,graph=True)


print(pagerank(models[6],qq,d,k,n,eps,maxiter))