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
        
        self.last_update_date = None
        self.curr_update_date = None
    
    def load_df(self, df):
        self.df = df
 
    def create_graph(self):
        #Get current datestamp with minutes and seconds
        current_datestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        #Add default values from the text files
        for index, row in self.df.iterrows():
            #For each row we first create an m4ml:MLModel instance
            model_uri = URIRef(f"mlentory:/hugging_face/{str(row['schema.org:name'][0]['data'])}")
            
            self.create_triplet(subject=model_uri, 
                                predicate=RDF.type, 
                                object=URIRef("fair4ml:MLModel"),
                                extraction_info=row['schema.org:name'][0])
            
            if(self.curr_update_date == None):
                self.curr_update_date = row['schema.org:name'][0]["extraction_time"]
            
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
            
            
            
    def create_triplet(self, subject, predicate, object, extraction_info):
        triplet_id = -1
        triplet_id_df = self.mySQLHandler.query(f"""SELECT id FROM Triplet WHERE subject = '{subject}' 
                                                                                     AND predicate = '{predicate}' 
                                                                                     AND object = '{object}'""")
        extraction_info_id = -1
        extraction_info_id_df = self.mySQLHandler.query(f"""SELECT id FROM Triplet_Extraction_Info WHERE 
                                                                    method_description = '{extraction_info["extraction_method"]}' 
                                                                    AND extraction_confidence = '{extraction_info["confidence"]}'""")
        
        
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
        
        if version_range_df.empty:
            #We have to create a new version range
            version_range_id = self.mySQLHandler.insert('Version_Range', {'triplet_id': str(triplet_id), 'extraction_info_id': str(extraction_info_id), 'start':extraction_info['extraction_time'], 'end':extraction_info['extraction_time'], 'deprecated':False})
        else:
            found_valid_range = False
            for index, row in version_range_df.iterrows():
                if row['end'] == self.last_update_date:
                    #Update the version range
                    found_valid_range = True
                    version_range_id = row['id']
                    self.mySQLHandler.update('Version_Range', {'end': extraction_info['extraction_time']}, f"id = '{row['id']}'")
                    break
            
            if not found_valid_range:
                #Create a new version range
                version_range_id = self.mySQLHandler.insert('Version_Range', {'triplet_id': triplet_id, 'extraction_info_id': extraction_info_id, 'start':extraction_info['extraction_time'], 'end':extraction_info['extraction_time'], 'deprecated':False})
        
        #All ranges that are not in this set will be deprecated
        # self.non_deprecated_ranges.add(version_range_id)
            
            
            
        # self.graph.add((subject, predicate, object))
        
        # if(extraction_info != None):
        #     # Create a new blank node to represent the extraction activity
        #     extraction_activity = BNode()
        #     # Add triplets to describe the extraction activity
        #     self.graph.add((extraction_activity, RDF.type, URIRef('prov:Activity')))
        #     # self.graph.add((extraction_activity, URIRef('prov:endedAtTime'), Literal(extraction_info['ended_at'])))
            
        #     extraction_method = extraction_info["extraction_method"]
        #     self.graph.add((subject, URIRef('prov:wasGeneratedBy'), extraction_activity))
        #     self.graph.add((extraction_activity, URIRef('prov:wasAssociatedWith'), URIRef(f"mlentory:extraction_methods/{extraction_method}")))
        #     self.graph.add((extraction_activity, URIRef('mlentory:extraction_confidence'), Literal(extraction_info['confidence'], datatype=XSD.float)))
        #     self.graph.add((extraction_activity, URIRef('prov:atTime'), Literal(extraction_info['extraction_time'], datatype=XSD.dateTime)))            
        #     self.graph.add((extraction_activity, URIRef('prov:generated'), object))
        #     self.graph.add((extraction_activity, URIRef('prov:generated'), predicate))

    def deprecate_old_triplets(self, model_uri):
        #First get the id of all the triplets related to the model
        update_query = f"""
            UPDATE Version_Range vr
            JOIN Triplet t ON t.id = vr.triplet_id
            SET vr.deprecated = 1, vr.end = '{self.curr_update_date}'
            WHERE t.subject = '{model_uri}'
            AND vr.end != '{self.curr_update_date}'
            AND vr.deprecated = 0
        """
        
        self.mySQLHandler.execute_sql(update_query)
        
        # model_triplets_with_ranges_df = self.mySQLHandler.query(f"""
        #     SELECT t.id AS triplet_id, vr.id AS range_id, vr.start, vr.end 
        #     FROM Triplet t
        #     JOIN Version_Range vr ON t.id = vr.triplet_id
        #     WHERE t.subject = '{model_uri}'
        #     AND vr.end != '{self.curr_update_date}'
        #     AND vr.deprecated = 0
        # """)
    
    def store_graph(self, file_path):
        self.graph.serialize(destination=file_path, format='turtle')
