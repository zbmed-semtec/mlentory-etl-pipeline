from elasticsearch import Elasticsearch
from elasticsearch_dsl import (
    Document,
    Field,
    Integer,
    Text,
    Nested,
    Date,
    Keyword,
    analyzer,
    tokenizer,
)


class Model(Document):
    """
    This class represents a model with all the FAIR4ML properties.
    """

    db_identifier = Text()
    
    name = Text(
        analyzer=analyzer(
            "title_analyzer",
            filter="lowercase",
            tokenizer=tokenizer("edge_ngram", "edge_ngram", min_gram=3, max_gram=10),
        ),
    )
    readme = Text()
    mlTask = Keyword(multi=True)
    author = Text(multi=True)
    
    
    # citation = Text()
    # version = Text()
    # ethicalLegalSocial = Text()
    
    # dateCreated = Date()
    # dateModified = Date()
    # datePublished = Date()

    
    # trainedOn = Text()
    # evaluatedOn = Text()
    # testedOn = Text()
    # fineTunedFrom = Text()

    # downloads = Integer()
    # storage_requirements = Integer()
    
    # license = Keyword()
    
    # mlTask = Keyword(multi=True)
    # softwareRequirements = Text(multi=True)
    # author = Text(multi=True)
    
    # sql_id = Text()
    # rdf_id = Text()

class HFModel(Model):
    """
    This class represents a model from Hugging Face Hub with its properties.
    """

    def save(self, **kwargs):
        return super(HFModel, self).save(**kwargs)
    
    class Meta:
        index = "hf_models"  # You can change the index name
        doc_type = "_doc"

    def upsert(self):
        dict_ = self.to_dict()
        dict_["_index"] = "hf_models"
        return dict_
    
    def props(self):   
        return [i for i in self.__dict__.keys() if i[:1] != '_']
    
