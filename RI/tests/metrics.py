
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.collection import Parser, IndexerSimple, QueryParser
from src.models import Weighter1, Weighter2, Weighter3, Weighter4, Weighter5, Vectoriel, ModeleLangue, Okapi
from src.metrics import optimisationModeleLangue, optimisationOkapi, EvalIRModel,crossValidationModeleLangue,crossValidationOkapi
from src.pagerank import pagerank, optimisationD
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

print("\n#################### Testing Model scores ####################\n")  
  


print("\n===================================\n\nTest of Vectoriel model with Weighter1 : ")
weighter = Weighter1(indexer)
model = Vectoriel(indexer,weighter,False)

eval_model(model,queries)

print("\n===================================\n\nTest of Vectoriel model with Weighter2 : ")
weighter = Weighter2(indexer)
model = Vectoriel(indexer,weighter,False)

eval_model(model,queries)

print("\n===================================\n\nTest of Vectoriel model with Weighter3 : ")
weighter = Weighter3(indexer)
model = Vectoriel(indexer,weighter,False)

eval_model(model,queries)

print("\n===================================\n\nTest of Vectoriel model with Weighter4 : ")
weighter = Weighter4(indexer)
model = Vectoriel(indexer,weighter,False)

eval_model(model,queries)


print("\n===================================\n\nTest of Vectoriel model with Weighter5 : ")
weighter = Weighter5(indexer)
model = Vectoriel(indexer,weighter,False)

eval_model(model,queries)

print("\n===================================\n\nTest of Language model with lamb = 0.7 : ")

model = ModeleLangue(indexer,0.7)

eval_model(model,queries)
print("\n===================================\n\nTest of Okapi BM25 model with k1 = 1.4 and b= 0.8: ")

model = Okapi(indexer,1.4,0.8)

eval_model(model,queries)
