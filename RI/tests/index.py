
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.collection import Parser, IndexerSimple, QueryParser






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

print("\nnombre de documents indexé : ")
print(len(indexer.index.keys()))

print("\nsaving the index...")

indexer.save("..\data\index")

print("Done...")


print("\nloading the index...")
indexer_loaded =  IndexerSimple()
indexer_loaded.load("..\data\index",docs)
print("Done...")

print("\nnombre de documents indexé : ")
print(len(indexer_loaded.index.keys()))

doc_id = '40'
print("\n\nindexed document id :",doc_id)
print(indexer.getTfsForDoc(doc_id))


stem = 'hand'
print("\n\nindexed stem :",stem)
print(indexer.getTfsForStem(stem))


print("\n\n TDIDF document id :",doc_id)
print(indexer.getTfIDFsForDoc(doc_id))


print("\n\n IFIDF stem :",stem)
print(indexer.getTfIDFsForStem(stem))
