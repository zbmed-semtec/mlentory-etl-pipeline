import numpy as np

np.float_ = np.float64
import pytest
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Search, Q, tokenizer


from elasticsearch_dsl import (
    Document,
    InnerDoc,
    Text,
    Integer,
    Keyword,
    Date,
    Nested,
    Boolean,
    analyzer,
    Index,
)


class TestElasticsearch:
    @pytest.fixture
    def es_client(self):
        # Connect to the Elasticsearch container
        # client = Elasticsearch(["http://elastic:9200"])
        client = Elasticsearch(
            [{"host": "elastic", "port": 9200, "scheme": "http"}],
            basic_auth=("elastic", "changeme"),
        )

        # print(client.info())
        yield client
        # Clean up all indexes after test
        self.clean_indices(client)

        # client.indices.refresh()

    @pytest.fixture
    def book_es_client(self, es_client):

        Book.init(index="book_index", using=es_client)

        author_1 = Author(name="Author1", age=30)
        author_2 = Author(name="Author2", age=40)

        docs = [
            Book(
                title="Book1 sdafdagfdagd",
                content="This is the content of book1",
                type=["Horror", "Suspense"],
                pages=100,
                author=author_1,
            ),
            Book(
                title="Book2",
                content="This is the content of book2",
                type=["Comedy", "Suspense"],
                pages=310,
                author=author_2,
            ),
            Book(
                title="Book3",
                content="This is the content of book3",
                type=["Comedy"],
                pages=265,
                author=author_2,
            ),
            Book(
                title="Book4",
                content="This is the content of book4",
                type=["Comedy", "Love"],
                pages=97,
                author=author_1,
            ),
        ]

        bulk(es_client, [element.upsert() for element in docs])
        es_client.indices.refresh(index="book_index")

        yield es_client

        self.clean_indices(es_client)

    def clean_indices(self, client):
        indices_to_delete = list(client.indices.get_alias(index="*").keys())
        if len(indices_to_delete) > 0:
            client.indices.delete(index=indices_to_delete, ignore=[400, 404])

    def test_add_document(self, es_client):
        doc = {"title": "Test Document", "content": "This is a test"}
        result = es_client.index(index="test_index", body=doc)
        assert result["result"] == "created"
        es_client.indices.delete(index=["test_index"], ignore=[400, 404])

    def test_modify_document(self, es_client):
        # First, add a document
        doc = {"title": "Original Title", "content": "Original content"}
        result = es_client.index(index="test_index", id="1", body=doc)

        # Now, modify the document
        updated_doc = {"title": "Updated Title", "content": "Updated content"}
        result = es_client.index(index="test_index", id="1", body=updated_doc)
        assert result["result"] == "updated"

    def test_query_document(self, es_client):
        # Add a document to query
        doc = {"title": "Queryable Document", "content": "This document can be queried"}
        es_client.index(index="test_index", id="2", body=doc)
        es_client.indices.refresh(index="test_index")

        # Perform a search
        result = es_client.search(
            index="test_index", body={"query": {"match": {"title": "Queryable"}}}
        )
        assert result["hits"]["total"]["value"] == 1
        assert result["hits"]["hits"][0]["_source"]["title"] == "Queryable Document"

    def test_bulk_insert(self, es_client):
        docs = [
            {
                "_index": "test_index",
                "_source": {"title": f"Bulk Doc {i}", "content": f"Bulk content {i}"},
            }
            for i in range(5)
        ]
        success, _ = bulk(es_client, docs)
        assert success == 5

    def test_create_mapping(self, es_client):
        body = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "keyword"},
                }
            }
        }

        es_client.indices.create(index="test_index", body=body)

        doc = {"title": "Test Document", "content": "Non-Fiction"}
        es_client.index(index="test_index", id="3", body=doc)
        es_client.indices.refresh(index="test_index")

        result = es_client.search(
            index="test_index", body={"query": {"match": {"title": "Test Docume"}}}
        )
        assert result["hits"]["total"]["value"] == 1

        result = es_client.search(
            index="test_index", body={"query": {"term": {"content": "Non-Fictio"}}}
        )
        assert result["hits"]["total"]["value"] == 0

        result = es_client.search(
            index="test_index", body={"query": {"term": {"content": "Non-Fiction"}}}
        )
        assert result["hits"]["total"]["value"] == 1

    def test_boolean_queries(self, book_es_client):
        s = (
            Search(using=book_es_client, index="book_index")
            .query("match_phrase", title="Book1")
            .filter("range", pages={"gt": 100})
        )

        result = s.execute()

        assert result["hits"]["total"]["value"] == 0

        s = (
            Search(using=book_es_client, index="book_index")
            .query("match_phrase", title="Book1")
            .filter("range", pages={"gte": 100})
        )
        result = s.execute()

        assert result["hits"]["total"]["value"] == 1

        s = (
            Search(using=book_es_client, index="book_index")
            .query("match_phrase", title="Book1")
            .filter("range", pages={"gt": 200})
        )
        result = s.execute()
        assert result["hits"]["total"]["value"] == 0

    def test_nested_queries(self, book_es_client):

        # self.print_all_documents(book_es_client)

        q = Q("nested", path="author", query=Q("match", author__name="Author2"))
        s = Search(using=book_es_client, index="book_index").query(q)

        result = s.execute()

        assert result["hits"]["total"]["value"] == 2

        # q = Q("nested", path="author", query=Q("match", author__name="Author2"))
        # s = Search(using=book_es_client, index="book_index").query(q)

        s = (
            Search(using=book_es_client, index="book_index")
            .query(Q("match", title="Boo"))
            .query(
                Q("nested", path="author", query=Q("range", author__age={"gte": 40}))
            )
            # .filter("range", author__age={"gte": 40})
        )

        result = s.execute()

        # print("INDEX Mapping: ", book_es_client.indices.get_mapping(index="book_index"))

        assert result["hits"]["total"]["value"] == 2

        s = (
            Search(using=book_es_client, index="book_index")
            .query("match", content="book1")
            .query(Q("nested", path="author", query=Q("match", author__name="Author2")))
        )
        result = s.execute()

        documents = result["hits"]["hits"]

        for doc in documents:
            # dict_keys(['_index', '_type', '_id', '_score', '_source'])

            print(doc["_type"])
            print(doc["_source"])

        assert result["hits"]["total"]["value"] == 0

    def test_tokenizers(self, book_es_client):

        self.print_all_documents(book_es_client)
        # print("INDEX Mapping: ", book_es_client.indices.get_mapping(index="book_index"))

        text_analyzer = analyzer(
            "my_tokenfilter",
            type="custom",
            tokenizer=tokenizer("trigram", "nGram", min_gram=3, max_gram=3),
        )
        response = text_analyzer.simulate("book1", using=book_es_client)
        tokens = [t.token for t in response.tokens]

        print("TEXT ANALYZER: ", tokens)
        # print("INDEX Mapping: ", Book.to_dict())
        print("INDEX Mapping: ", book_es_client.indices.get_mapping(index="book_index"))

    def test_delete_document(self, es_client):
        # Add a document to delete
        doc = {"title": "Delete Me", "content": "This document will be deleted"}
        es_client.index(index="test_index", id="3", body=doc)

        # Delete the document
        result = es_client.delete(index="test_index", id="3")
        assert result["result"] == "deleted"

        # Verify deletion
        with pytest.raises(Exception):
            es_client.get(index="test_index", id="3")

    def print_all_documents(self, client):
        # Perform a search that matches all documents
        response = client.search()

        # Access the list of documents from the response
        documents = response["hits"]["hits"]

        # Print the documents
        print("Printing all documents:")
        for doc in documents:
            # dict_keys(['_index', '_type', '_id', '_score', '_source'])

            print(doc["_type"])
            print(doc["_source"])
            print(
                "TERM vectors: ",
                client.termvectors(index="book_index", id=doc["_id"], fields="title"),
            )


class Author(InnerDoc):
    name = Text(
        analyzer=analyzer(
            "title_analyzer",
            filter="lowercase",
            tokenizer=tokenizer("edge_ngram", "edge_ngram", min_gram=3, max_gram=10),
        ),
    )
    age = Integer()

    class Meta:
        index = "book_index"
        doc_type = "_doc"


class Book(Document):
    title = Text(
        analyzer=analyzer(
            "title_analyzer",
            filter="lowercase",
            tokenizer=tokenizer("edge_ngram", "edge_ngram", min_gram=3, max_gram=10),
        ),
    )
    content = Text()
    pages = Integer()
    category = Keyword()

    author = Nested(Author())

    def save(self, **kwargs):
        self.lines = len(self.body.split())
        return super(Book, self).save(**kwargs)

    def upsert(self):
        dict_ = self.to_dict()
        dict_["_index"] = "book_index"
        return dict_

    class Meta:
        index = "book_index"
        doc_type = "_doc"
