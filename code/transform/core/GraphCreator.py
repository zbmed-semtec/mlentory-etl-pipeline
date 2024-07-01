import rdflib
from rdflib import URIRef

class GraphCreator:
    def __init__(self):
        self.df_to_transform = None
        self.graph = rdflib.Graph()
        self.graph.bind('fair4ml', URIRef('http://fair4ml.com/'))
        self.graph.bind('codemeta', URIRef('http://codemeta.com/'))
        self.graph.bind('schema', URIRef('https://schema.org/'))
        #Now I want to create a node that represents the list of models
        self.create_triplet(subject=URIRef(''), predicate="instanceOf", object=URIRef("m4ml:HFModelList"))
        
    
    def load_df(self, df):
        self.df = df
        
    def create_graph(self):
        for index, row in self.df.iterrows():
            #For each row we first create an m4ml:MLModel instance
            self.create_triplet(subject=URIRef(''), predicate="instanceOf", object=URIRef("m4ml:Model"))
            print("CHECKING: ",row['schema.org:name'])
            self.create_triplet(subject=row['schema.org:name'], predicate="fair4ml:evaluatedOn", object=row['fair4ml:evaluatedOn'])
            self.create_triplet(subject=row['schema.org:name'], predicate="fair4ml:mlTask", object=row['fair4ml:mlTask'])
            self.create_triplet(subject=row['schema.org:name'], predicate="fair4ml:sharedBy", object=row['fair4ml:sharedBy'])
            self.create_triplet(subject=row['schema.org:name'], predicate="fair4ml:testedOn", object=row['fair4ml:testedOn'])
            self.create_triplet(subject=row['schema.org:name'], predicate="codemeta:referencePublication", object=row['codemeta:referencePublication'])
            

    def create_triplet(self, subject, predicate, object):
        rdf_subject = rdflib.URIRef(subject)
        rdf_predicate = rdflib.URIRef(predicate)
        
        rdf_object = rdflib.Literal(object)
        self.graph.add((rdf_subject, rdf_predicate, rdf_object))
    
    def store_graph(self, file_path):
        self.graph.serialize(destination=file_path, format='turtle')
