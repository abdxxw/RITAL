
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.collection import Parser, IndexerSimple, QueryParser
from src.models import ModeleLangue, Okapi
from src.metrics import optimisationModeleLangue, optimisationOkapi, EvalIRModel,crossValidationModeleLangue,crossValidationOkapi
from src.TextRepresenter import PorterStemmer
import random
import itertools 





# =================== parser and index =================================

print("\n#################### PARSING ###################\n")
parser = Parser()
docs = parser.parse("..\data\cacm\cacm.txt")

qParser = QueryParser()
queries = qParser.parse("..\data\cacm\cacm.qry","..\data\cacm\cacm.rel")

print("Done...")

print("\n#################### Indexing ###################\n")
indexer =  IndexerSimple()
indexer.indexation(docs)

print("Done...")



def score_query(q):
    ps = PorterStemmer()
    return ps.getTextRepresentation(q)

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




print("\n#################### GridSearch ####################\n")  

trainDict , testDict = split(queries,0.8)


startlamb, endlamb, steplamb = 0.1, 1, 0.1

startk, endk, stepk = 1, 2, 0.1

startb, sendb, stepb = 0.1, 1, 0.1

print("\n===================================\n\n ModeleLangue Searching for best lamb... ")

lamb = optimisationModeleLangue(indexer,startlamb,endlamb,steplamb,trainDict,graph=True)

print("Best lamb : "+str(lamb))

m = ModeleLangue(indexer,lamb)

eval_model(m,testDict)



print("\n===================================\n\n BM25 Searching for best k1 and b... ")

k1 , b = optimisationOkapi(indexer,startk, endk, stepk, startb, sendb, stepb,trainDict,graph=True)

print("Best params k1, b : "+str(k1)+", "+str(b))

m = Okapi(indexer,k1,b)

eval_model(m,testDict)


print("\n#################### Cross Validation (takes time) ####################\n")   


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

