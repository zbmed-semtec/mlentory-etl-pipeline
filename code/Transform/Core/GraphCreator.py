import rdflib

class GraphCreator:
    def __init__(self):
        self.df_to_transform = None
        self.graph = rdflib.Graph()
    
    def load_df(self, df):
        self.df = df
        
    def create_graph(self):
        for index, row in self.df.iterrows():
            #For each row we first create an m4ml:MLModel instance
            self.create_triplet(subject="m4ml:Model", predicate="instanceOf", object=row['schema.org:name'])
            

    def create_triplet(self, subject, predicate, object):
        rdf_subject = rdflib.URIRef(subject)
        rdf_predicate = rdflib.URIRef(predicate)
        rdf_object = rdflib.Literal(object)
        self.graph.add((rdf_subject, rdf_predicate, rdf_object))
    
    def store_graph(self, file_path):
        self.graph.serialize(destination=file_path, format='turtle')
