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
        self.graph.bind('prov', URIRef('http://www.w3.org/ns/prov#'))
        #Now I want to create a node that represents the list of models
        
    
    def load_df(self, df):
        self.df = df
 
    def create_graph(self):
        #Add default values from the text files
        for index, row in self.df.iterrows():
            #For each row we first create an m4ml:MLModel instance
            model_uri = URIRef(f"mlentory:hugging_face:{str(row['schema.org:name'][0]['data'])}")
            self.create_triplet(subject=model_uri, predicate=RDF.type, object=URIRef("fair4ml:MLModel"))
            self.create_triplet(subject=model_uri, predicate="fair4ml:evaluatedOn", object=row['fair4ml:evaluatedOn'][0]["data"], extraction_info=row['fair4ml:evaluatedOn'][0])
            self.create_triplet(subject=model_uri, predicate="fair4ml:mlTask", object=row['fair4ml:mlTask'][0]["data"],extraction_info=row['fair4ml:evaluatedOn'][0])
            self.create_triplet(subject=model_uri, predicate="fair4ml:sharedBy", object=row['fair4ml:sharedBy'][0]["data"],extraction_info=row['fair4ml:evaluatedOn'][0])
            self.create_triplet(subject=model_uri, predicate="fair4ml:testedOn", object=row['fair4ml:testedOn'][0]["data"], extraction_info=row['fair4ml:testedOn'][0])
            self.create_triplet(subject=model_uri, predicate="codemeta:referencePublication", object=row['codemeta:referencePublication'][0]["data"], extraction_info=row['codemeta:referencePublication'][0])
            
            
            

    def create_triplet(self, subject, predicate, object, extraction_info=None):
        rdf_subject = rdflib.URIRef(subject)
        rdf_predicate = rdflib.URIRef(predicate)
        rdf_object = rdflib.Literal(object)
        self.graph.add((rdf_subject, rdf_predicate, rdf_object))
        
        if(extraction_info != None):
            # Create a new blank node to represent the extraction activity
            extraction_activity = BNode()
            # Add triplets to describe the extraction activity
            self.graph.add((extraction_activity, RDF.type, URIRef('prov:Activity')))
            # self.graph.add((extraction_activity, URIRef('prov:endedAtTime'), Literal(extraction_info['ended_at'])))
            print("HERREEEEE ",extraction_info)
            extraction_method = extraction_info["extraction_method"]
            self.graph.add((extraction_activity, URIRef('prov:wasAssociatedWith'), URIRef(f"mlentory:extraction_methods:{extraction_method}")))
            self.graph.add((extraction_activity, URIRef('mlentory:extraction_confidence'), Literal(extraction_info['confidence'])))
            self.graph.add((rdf_object, URIRef('prov:wasGeneratedBy'), extraction_activity))
            

    
    def store_graph(self, file_path):
        self.graph.serialize(destination=file_path, format='turtle')
