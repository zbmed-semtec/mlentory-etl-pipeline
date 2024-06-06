import rdflib

class GraphCreator:
    def __init__(self, df):
        self.df = df
        self.graph = rdflib.Graph()

    def create_graph(self):
        for index, row in self.df.iterrows():
            subject = rdflib.URIRef(row['subject'])
            predicate = rdflib.URIRef(row['predicate'])
            object = rdflib.Literal(row['object'])
            self.graph.add((subject, predicate, object))

    def store_graph(self, file_path):
        self.graph.serialize(destination=file_path, format='turtle')
