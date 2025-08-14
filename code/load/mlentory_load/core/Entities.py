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

    db_identifier = Keyword()

    name = Text(
        analyzer=analyzer(
            "title_analyzer",
            filter="lowercase",
            tokenizer=tokenizer("edge_ngram", "edge_ngram", min_gram=3, max_gram=30),
        ),
    )
    description = Text()
    license = Keyword()
    sharedBy = Text()
    mlTask = Keyword(multi=True)
    keywords = Keyword(multi=True)  
    relatedDatasets = Text(multi=True)
    baseModels = Text(multi=True)
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

    class Meta:
        index = "hf_models"  # You can change the index name
        doc_type = "_doc"

    def __init__(self, **kwargs):
        # self.meta.index = index
        super(HFModel, self).__init__(**kwargs)

    def save(self, **kwargs):
        return super(HFModel, self).save(**kwargs)

    def upsert(self):
        dict_ = self.to_dict()
        dict_["_index"] = self.meta.index
        return dict_

    def props(self):
        return [i for i in self.__dict__.keys() if i[:1] != "_"]
    
class Run(Document):
    """
    This class represents a run with all the FAIR4ML properties.
    """

    db_identifier = Keyword()

    name = Text(
        analyzer=analyzer(
            "title_analyzer",
            filter="lowercase",
            tokenizer=tokenizer("edge_ngram", "edge_ngram", min_gram=3, max_gram=30),
        ),
    )
    license = Keyword()
    mlTask = Keyword(multi=True)
    sharedBy = Text()
    modelCategory = Keyword(multi=True)
    trainedOn = Text(multi=True)
    keywords = Keyword(multi=True) 

class OpenMLRun(Run):
    """
    This class represents a run from OpenML platform with its properties.
    """

    class Meta:
        index = "openml_models"
        doc_type = "_doc"

    def __init__(self, **kwargs):
        super(OpenMLRun, self).__init__(**kwargs)

    def save(self, **kwargs):
        return super(OpenMLRun, self).save(**kwargs)

    def upsert(self):
        dict_ = self.to_dict()
        dict_["_index"] = self.meta.index
        return dict_

    def props(self):
        return [i for i in self.__dict__.keys() if i[:1] != "_"]