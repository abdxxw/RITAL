

import TextRepresenter as TR
import numpy as np
import json
import re

###########################		TME1		################################ 






class Document:
    def __init__(self, id, titre = None, auteur = None, date = None, keys = None, text = None, linkTo=dict(),linkFrom=[]):
        self.id = id
        self.titre = titre
        self.date = date
        self.auteur = auteur
        self.keys = keys
        self.text = text
        self.linkTo = linkTo
        self.linkFrom=linkFrom
        
    def getID(self):
        return self.id

    def getText(self):
        return self.text
    
    def setLinkFrom(self, l):
        self.linkFrom = l

    def getHyperlinksTo(self):
        return self.linkTo
    
    def getHyperlinksFrom(self):
        return self.linkFrom


class Parser:
    
    def __init__(self):
        self.documents = dict()
       
    def parse(self,fichier):
        
        docs = open(fichier, "r").read().split('.I ')
        allLinkFrom = dict()
         
        for i in range(1,len(docs)):
            linkTo = dict()
            idDoc = re.search(r"[0-9]+",docs[i])
            if idDoc is not None:
                idDoc = idDoc.group(0)
            title = self.get_element("T",docs[i])
            date = self.get_element("B",docs[i])
            auteur = self.get_element("A",docs[i])
            text = self.get_element("W",docs[i])
            keys = self.get_element("K",docs[i])
            links = self.get_element("X",docs[i])
            
            if links is not None:
                l = links.split("\n")
                for i in range(1,len(l)):
                    linkID = l[i].split()[0]
                    linkTo[int(linkID)] = linkTo.get(int(linkID), 0) + 1
                    
                    allLinkFrom[linkID] = allLinkFrom.get(linkID, []) + [idDoc]
                    
            d = Document(idDoc,title,auteur,date,keys,text,linkTo)
            self.documents[idDoc]=d 
            
        for idDoc in allLinkFrom:
            self.documents[idDoc].setLinkFrom(allLinkFrom[idDoc])
                
        return self.documents

    def get_element(self,balise,doc):
        res = re.search(r"\."+balise+"([\s\S]*?)\.[ITBAKNWX]",doc)
        if res is not None:
            return res.group(1)
        else:
            res = re.search(r"\."+balise+"([\s\S]*?)\n\n",doc)
            if res is not None:
                return res.group(1)
            else:
                return None   
    
class Query():
    
    def __init__(self, id, text, pertinences):
        
        self.id = id
        self.txt = text
        self.listPert = pertinences 

        
    def getID(self):
        return self.id

    def getText(self):
        return self.txt
    
    
class QueryParser():
    
    def __init__(self):
        
        self.queries = dict()  

       
    def parse(self,fichierQ,fichierQrels):
        
        with open(fichierQrels) as fqrels:
            lines = fqrels.readlines()
            qrels = dict()
            
            for l in lines: 
                tmp = l.split()
                if(tmp[0] in qrels.keys()):
                    qrels[tmp[0]].append(tmp[1])
                else:
                    qrels[tmp[0]] = [tmp[1]]
                    
        docs = open(fichierQ, "r").read().split('.I ')

        for i in range(1,len(docs)):

            idQ = re.search(r"[0-9]+",docs[i])
            if idQ is not None:
                idQ = idQ.group(0)
                
            text = self.get_element("W",docs[i])


            pert = qrels.get(idQ,[])        
            q = Query(idQ,text,pert)
            self.queries[idQ]=q

        return self.queries

    def get_element(self,balise,doc):
        res = re.search(r"\."+balise+"([\s\S]*?)\.[ITBAKNWX]",doc)
        if res is not None:
            return res.group(1)
        else:
            res = re.search(r"\."+balise+"([\s\S]*)\n",doc)
            if res is not None:
                return res.group(1)
            else:
                return None   

        

class IndexerSimple():
    
    def __init__(self):

        self.index = dict()
        self.indexInv = dict()
        
            
    def getIndex(self):
        return self.index

    def getIndexInv(self):
        return self.indexInv
    
    def indexation(self,dictDoc):  
        
        self.dictDoc = dictDoc
        Trep=TR.PorterStemmer()
        for i in dictDoc.keys():   
            txt = dictDoc[i].getText()
            if txt is not None:
                terms= Trep.getTextRepresentation(dictDoc[i].getText())
                self.index[i] = dict(terms)
            else:
                self.index[i] = dict()
            
            for t in self.index[i].keys():
                if (t in self.indexInv.keys()):
                    if(i in self.indexInv[t].keys()):
                        self.indexInv[t][i] += self.index[i][t]
                    else:
                        self.indexInv[t][i]= self.index[i][t]
                else:
                    self.indexInv[t] = {}
                    self.indexInv[t][i]=self.index[i][t]
    
    def save(self,IndexFile,IndexInvFile):
        with open(IndexFile,'w') as f:
            json.dump(self.index, f)
        with open(IndexInvFile,'w') as f:
            json.dump(self.indexInv, f)
        
    def load(self,IndexFile,IndexInvFile):
        with open(IndexFile,'r') as f:
            self.index = json.load(f)
        with open(IndexFile,'r') as f:
            self.indexInv = json.load(f)
        
        
    def getTfsForDoc(self,doc):
        return self.index[doc]
    
    
    
    def getTfIDFsForDoc(self,doc):
        N=len(self.index)
        dic = dict()
        for t in self.index[doc].keys():
            dic[t] = self.index[doc][t] * np.log((1+N) /(1+ len(self.indexInv[t])))
        return dic
    
    def getTfsForStem(self,stem):
        return self.indexInv[stem]
    
    def getTfIDFsForStem(self,stem):
        
        N = len(self.index)
        tfidf = dict()
        for t in self.indexInv[stem].keys():
            tfidf[t] = self.index[t][stem] * np.log((1+N)/(1 + len(self.indexInv[stem])))
        return tfidf
    
    def getIDFsForStem(self,stem):
        N = len(self.index)
        if stem not in self.indexInv.keys():
            return 0
        return np.log((1+N)/(1 + len(self.indexInv[stem])))
    
    def getStrDoc(self,parser,doc):
        return parser.documents[doc].txt
    
    def getCollection(self):   
        return self.dictDoc
		
    def getCollectionSize(self):
        return sum(self.getDocumentSize(k) for k in self.index.keys())
    
    def getDocumentSize(self,idDoc):
        return sum(self.index[idDoc].values())


    def getHyperlinksTo(self,doc):
        return self.dictDoc[doc].getHyperlinksTo()
    
    def getHyperlinksFrom(self,doc):
        return self.dictDoc[doc].getHyperlinksFrom()

    
###########################		TME3		################################ 
