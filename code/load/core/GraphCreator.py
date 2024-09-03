import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, XSD, FOAF
import uuid
import json
import pandas as pd
import os
from datetime import datetime
from typing import Callable, List, Dict,Set

if("app_test" in os.getcwd()):
    from load.core.dbHandler.MySQLHandler import MySQLHandler
    from load.core.dbHandler.VirtuosoHandler import VirtuosoHandler
else:
    from core.dbHandler.MySQLHandler import MySQLHandler
    from core.dbHandler.VirtuosoHandler import VirtuosoHandler

class GraphCreator:
    def __init__(self, mySQLHandler:MySQLHandler, virtuosoHandler:VirtuosoHandler):
        self.df_to_transform = None
        self.graph = rdflib.Graph()
        self.mySQLHandler = mySQLHandler
        self.virtuosoHandler = virtuosoHandler
        self.graph.bind('fair4ml', URIRef('http://fair4ml.com/'))
        self.graph.bind('codemeta', URIRef('http://codemeta.com/'))
        self.graph.bind('schema', URIRef('https://schema.org/'))
        self.graph.bind('mlentory', URIRef('https://mlentory.com/'))
        self.graph.bind('prov', URIRef('http://www.w3.org/ns/prov#'))
        
        self.non_deprecated_ranges = set()
        
        # self.last_update_date = None
        self.curr_update_date = None
    
    def load_df(self, df):
        self.df = df
 
    def create_graph(self):
        
        for index, row in self.df.iterrows():
            #For each row we first create an m4ml:MLModel instance
            model_uri = URIRef(f"mlentory:/hugging_face/{str(row['schema.org:name'][0]['data'])}")
            
            self.create_triplet(subject=model_uri, 
                                predicate=RDF.type, 
                                object=URIRef("fair4ml:MLModel"),
                                extraction_info=row['schema.org:name'][0])
            
            if(self.curr_update_date == None):
                self.curr_update_date = datetime.strptime(row['schema.org:name'][0]["extraction_time"], '%Y-%m-%d_%H-%M-%S')
            
            #Go through all the columns and add the triplets
            for column in self.df.columns:
                if(column == 'schema.org:name'):
                    continue
                #Handle the cases where a new entity has to be created
                if(column in ["fair4ml:mlTask", "fair4ml:sharedBy", "fair4ml:testedOn", "fair4ml:trainedOn","codemeta:referencePublication"]):
                    # Go through the different sources that can create information about the entity
                    if(type(row[column])!=list and pd.isna(row[column])):
                        continue
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
                                                        object=URIRef(f"mlentory:hugging_face/{data.replace(' ','_')}"),
                                                        extraction_info=source)
                        elif type(source["data"]) == list:
                            for entity in source["data"]:
                                self.create_triplet(subject=model_uri,
                                                            predicate=rdflib.URIRef(column),
                                                            object=URIRef(f"mlentory:hugging_face/{entity.replace(' ','_')}"),
                                                            extraction_info=source)
                if(column in ["dateCreated", "dateModified"]):
                    self.create_triplet(subject=model_uri,
                                        predicate=rdflib.URIRef(column),
                                        object=Literal(row[column]["data"], datatype=XSD.date))
                if(column in ["storageRequirements","name"]):
                    self.create_triplet(subject=model_uri,
                                        predicate=rdflib.URIRef(column),
                                        object=Literal(row[column]["data"], datatype=XSD.string))
            
            #Deprecate all the triplets that were not created or updated for the current model
            self.deprecate_old_triplets(model_uri)
        
        self.update_triplet_ranges_for_unchanged_models(self.curr_update_date)
        self.curr_update_date = None
            
            
            
    def create_triplet(self, subject, predicate, object, extraction_info):
        triplet_id = -1
        triplet_id_df = self.mySQLHandler.query(f"""SELECT id FROM Triplet WHERE subject = '{subject}' 
                                                                                     AND predicate = '{predicate}' 
                                                                                     AND object = '{object}'""")
        extraction_info_id = -1
        extraction_info_id_df = self.mySQLHandler.query(f"""SELECT id FROM Triplet_Extraction_Info WHERE 
                                                                    method_description = '{extraction_info["extraction_method"]}' 
                                                                    AND extraction_confidence = {extraction_info["confidence"]}""")
        
        
        # print(f"result_triple_exist: {triplet_id}")
        
        if triplet_id_df.empty:
            #We have to create a new triplet
            # print("Triplet not found in the database")
            triplet_id = self.mySQLHandler.insert('Triplet', {'subject': subject, 'predicate': predicate, 'object': object})
        else:
            triplet_id = triplet_id_df.iloc[0]['id']
        
        if extraction_info_id_df.empty:
            #We have to create a new extraction info
            # print("Extraction info not found in the database")
            extraction_info_id = self.mySQLHandler.insert('Triplet_Extraction_Info', {'method_description': extraction_info["extraction_method"], 'extraction_confidence': extraction_info["confidence"]})
        else:
            extraction_info_id = extraction_info_id_df.iloc[0]['id']
            
            
        #We already have the triplet and the extraction info
        #We need to check the version_extraction_range 
        version_range_df = self.mySQLHandler.query(f"""SELECT id,start,end FROM Version_Range WHERE
                                                                        triplet_id = '{triplet_id}'
                                                                        AND extraction_info_id = '{extraction_info_id}'
                                                                        AND deprecated = {False}""")
        version_range_id = -1
        extraction_time = datetime.strptime(extraction_info['extraction_time'], '%Y-%m-%d_%H-%M-%S')
        
        if version_range_df.empty:
            #We have to create a new version range
            version_range_id = self.mySQLHandler.insert('Version_Range', {'triplet_id': str(triplet_id), 'extraction_info_id': str(extraction_info_id), 'start':extraction_time, 'end':extraction_time, 'deprecated':False})
        
        else:
            version_range_id = version_range_df.iloc[0]['id']
            self.mySQLHandler.update('Version_Range', {'end': extraction_time}, f"id = '{version_range_id}'")

    def deprecate_old_triplets(self, model_uri):
        
        update_query = f"""
            UPDATE Version_Range vr
            JOIN Triplet t ON t.id = vr.triplet_id
            SET vr.deprecated = 1, vr.end = '{self.curr_update_date}'
            WHERE t.subject = '{model_uri}'
            AND vr.end < '{self.curr_update_date}'
            AND vr.deprecated = 0
        """
        self.mySQLHandler.execute_sql(update_query)
        
    
    def update_triplet_ranges_for_unchanged_models(self, curr_date:datetime) -> None:
        """
        The idea is to update all the triplet ranges that were not modified in the last update
        to have the same end date as the current date.
        """
        
        update_query = f"""
            UPDATE Version_Range vr
            SET vr.end = '{curr_date}'
            WHERE vr.end != '{curr_date}'
            AND vr.deprecated = 0
        """
        
        self.mySQLHandler.execute_sql(update_query)
        
    
    def store_graph(self, file_path):
        self.graph.serialize(destination=file_path, format='turtle')
