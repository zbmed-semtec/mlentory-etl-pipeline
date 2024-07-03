import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF
import uuid
import json
import pandas as pd

class GraphCreator:
    def __init__(self):
        self.df_to_transform = None
        self.graph = rdflib.Graph()
        self.graph.bind('fair4ml', URIRef('http://fair4ml.com/'))
        self.graph.bind('codemeta', URIRef('http://codemeta.com/'))
        self.graph.bind('schema', URIRef('https://schema.org/'))
        self.graph.bind('mlentory', URIRef('https://mlentory.com/'))
        #Now I want to create a node that represents the list of models
        
    
    def load_df(self, df):
        self.df = df
        # for col in self.df.columns:
        #     # self.df[col] = self.df[col].apply(lambda x: str(x).replace('{\'data\': \'', '{\"data\": \"')
        #     #                                                   .replace('{\'data\': [', '{\"data\": [')
        #     #                                                   .replace('\', \'extraction_method\': \'', '\", \"extraction_method\": \"')
        #     #                                                   .replace('], \'extraction_method\': \'', '], \"extraction_method\": \"')
        #     #                                                   .replace('\', \'confidence\':', '\", \"confidence\":'))
        #     self.df[col] = self.df[col].apply(lambda x: eval(x) if not pd.isna(x) else x)

        
    def create_graph(self):
        for index, row in self.df.iterrows():
            #For each row we first create an m4ml:MLModel instance
            model_uri = BNode()
            self.create_triplet(subject=model_uri, predicate=RDF.type, object=URIRef("m4ml:MLModel"))
            print("CHECKING: ",row['schema.org:name'])
            self.create_triplet(subject=model_uri, predicate="fair4ml:evaluatedOn", object=Literal(row['fair4ml:evaluatedOn'][0]["data"]))
            self.create_triplet(subject=model_uri, predicate="fair4ml:mlTask", object=Literal(row['fair4ml:mlTask'][0]["data"]))
            self.create_triplet(subject=model_uri, predicate="fair4ml:sharedBy", object=Literal(row['fair4ml:sharedBy'][0]["data"]))
            self.create_triplet(subject=model_uri, predicate="fair4ml:testedOn", object=Literal(row['fair4ml:testedOn'][0]["data"]))
            self.create_triplet(subject=model_uri, predicate="codemeta:referencePublication", object=Literal(row['codemeta:referencePublication'][0]["data"]))
            

    def create_triplet(self, subject, predicate, object):
        rdf_subject = rdflib.URIRef(subject)
        rdf_predicate = rdflib.URIRef(predicate)
        
        rdf_object = rdflib.Literal(object)
        self.graph.add((rdf_subject, rdf_predicate, rdf_object))
    
    def store_graph(self, file_path):
        self.graph.serialize(destination=file_path, format='turtle')
