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
