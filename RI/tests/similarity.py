
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.collection import Parser, IndexerSimple, QueryParser
from src.models import Weighter1, Weighter2, Weighter3, Weighter4, Weighter5, Vectoriel, ModeleLangue, Okapi
from src.metrics import EvalIRModel

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



print("\n===================================\n\n Genereating models similarity ... ")

    

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


print("Done...")
