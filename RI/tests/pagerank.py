
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.collection import Parser, IndexerSimple, QueryParser
from src.models import Weighter1, Weighter2, Weighter3, Weighter4, Weighter5, Vectoriel, ModeleLangue, Okapi
from src.pagerank import pagerank, optimisationD
from src.TextRepresenter import PorterStemmer





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


start1, end1, n1 = 0.1, 1, 0.1
k=2
n=2
eps = 0.1
maxiter = 1000

id_q = 31
q = queries[list(queries.keys())[id_q]].getText()
query = score_query(q)

print("\n===================================\n\nTest of Vectoriel model with Weighter1 : ")
weighter = Weighter1(indexer)
model = Vectoriel(indexer,weighter,False)

d = optimisationD(model,start1,end1,n1,query,k,n,eps,maxiter,graph=False)

print(pagerank(model,query,d,k,n,eps,maxiter))

print("\n===================================\n\nTest of Vectoriel model with Weighter2 : ")
weighter = Weighter2(indexer)
model = Vectoriel(indexer,weighter,False)

d = optimisationD(model,start1,end1,n1,query,k,n,eps,maxiter,graph=False)

print(pagerank(model,query,d,k,n,eps,maxiter))

print("\n===================================\n\nTest of Vectoriel model with Weighter3 : ")
weighter = Weighter3(indexer)
model = Vectoriel(indexer,weighter,False)

d = optimisationD(model,start1,end1,n1,query,k,n,eps,maxiter,graph=False)

print(pagerank(model,query,d,k,n,eps,maxiter))

print("\n===================================\n\nTest of Vectoriel model with Weighter4 : ")
weighter = Weighter4(indexer)
model = Vectoriel(indexer,weighter,False)

d = optimisationD(model,start1,end1,n1,query,k,n,eps,maxiter,graph=False)

print(pagerank(model,query,d,k,n,eps,maxiter))


print("\n===================================\n\nTest of Vectoriel model with Weighter5 : ")
weighter = Weighter5(indexer)
model = Vectoriel(indexer,weighter,False)

d = optimisationD(model,start1,end1,n1,query,k,n,eps,maxiter,graph=False)

print(pagerank(model,query,d,k,n,eps,maxiter))

print("\n===================================\n\nTest of Language model with lamb = 0.7 : ")

model = ModeleLangue(indexer,0.7)

d = optimisationD(model,start1,end1,n1,query,k,n,eps,maxiter,graph=False)

print(pagerank(model,query,d,k,n,eps,maxiter))
print("\n===================================\n\nTest of Okapi BM25 model with k1 = 1.4 and b= 0.8: ")

model = Okapi(indexer,1.4,0.8)

d = optimisationD(model,start1,end1,n1,query,k,n,eps,maxiter,graph=False)

print(pagerank(model,query,d,k,n,eps,maxiter))






