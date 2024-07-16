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
            
            self.create_triplet(subject=model_uri, 
                                predicate=RDF.type, 
                                object=URIRef("fair4ml:MLModel"))
            
            #Go through all the columns and add the triplets
            for column in self.df.columns:
                if(column == 'schema.org:name'):
                    continue
                #Handle the cases where a new entity has to be created
                if(column in ["fair4ml:mlTask", "fair4ml:sharedBy", "fair4ml:testedOn", "fair4ml:trainedOn","codemeta:referencePublication"]):
                    # Go through the different sources that can create information about the entity
                    for source in row[column]:
                        if source["data"] == None:
                            self.create_triplet(subject=model_uri,
                                                        predicate= rdflib.URIRef(column),
                                                        object=Literal("None"),
                                                        extraction_info=source)
                        elif type(source["data"]) == str:
                            data = source["data"]
                            self.create_triplet(subject=model_uri,
                                                        predicate=rdflib.URIRef(column),
                                                        object=URIRef(f"mlentory:hugging_face:{data.replace(' ','_')}"),
                                                        extraction_info=source)
                        elif type(source["data"]) == list:
                            for entity in source["data"]:
                                self.create_triplet(subject=model_uri,
                                                            predicate=rdflib.URIRef(column),
                                                            object=URIRef(f"mlentory:hugging_face:{entity.replace(' ','_')}"),
                                                            extraction_info=source)
            
            
            
    def create_triplet(self, subject, predicate, object, extraction_info=None):
        self.graph.add((subject, predicate, object))
        
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
            self.graph.add((extraction_activity, URIRef('prov:atTime'), Literal(extraction_info['extraction_time'])))
            self.graph.add((extraction_activity, URIRef('prov:generated'), subject))
            self.graph.add((extraction_activity, URIRef('prov:generated'), object))
            self.graph.add((extraction_activity, URIRef('prov:generated'), predicate))

            

    
    def store_graph(self, file_path):
        self.graph.serialize(destination=file_path, format='turtle')
