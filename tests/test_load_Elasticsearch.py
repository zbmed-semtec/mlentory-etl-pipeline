# Don't change the following line.
import numpy as np
np.float_ = np.float64
import pytest
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class TestElasticsearch:
    @pytest.fixture
    def es_client(self):
        # Connect to the Elasticsearch container
        # client = Elasticsearch(["http://elastic:9200"])
        client = Elasticsearch(
            [{"host":"elastic", "port":9200, "scheme": "http"}],
            basic_auth=('elastic', 'changeme')
        )
        
        # print(client.info())
        yield client
        # Clean up all indexes after test
        self.clean_indices(client)
        
        
        # client.indices.refresh()
    
    @pytest.fixture
    def book_es_client(self, es_client):
        # Create an index
        # Create a mapping
        
        body = {"mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "type": {"type": "keyword"},
                    "number of pages": {"type": "integer"}
                }
            }
        }
        es_client.indices.create(index="book_index", body= body)
        
        docs = [{"_index":"book_index","title": "Book1", "content": "This is the content of book1","number of pages": 100},
                {"_index":"book_index","title": "Book2", "content": "This is the content of book2","number of pages": 310},
                {"_index":"book_index","title": "Book3", "content": "This is the content of book3","number of pages": 265},
                {"_index":"book_index","title": "Book4", "content": "This is the content of book4","number of pages": 97}]
        
        bulk(es_client, docs)
        es_client.indices.refresh(index="book_index")
        
        yield es_client
        
        self.clean_indices(es_client)

    @pytest.fixture
    def book_nested_es_client(self, es_client):
        # Create an index
        # Create a mapping
        body = {"mapes_client.indices.delete(index=['test_index'], ignore=[400, 404])pings": {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "type": {"type": "keyword"},
                    "number of pages": {"type": "integer"}
                }
            }
        }
        es_client.indices.create(index="book_index", body= body)
        
        docs = [{"_index":"book_index","title": "Book1", "content": "This is the content of book1","number of pages": 100},
                {"_index":"book_index","title": "Book2", "content": "This is the content of book2","number of pages": 310},
                {"_index":"book_index","title": "Book3", "content": "This is the content of book3","number of pages": 265},
                {"_index":"book_index","title": "Book4", "content": "This is the content of book4","number of pages": 97}]
        
        bulk(es_client, docs)
        es_client.indices.refresh(index="book_index")
        
        yield es_client
        
        self.clean_indices(es_client)

    def clean_indices(self, client):
        indices_to_delete = list(client.indices.get_alias(index="*").keys())
        if len(indices_to_delete) > 0:
            client.indices.delete(index=indices_to_delete, ignore=[400, 404])
        print("After delete: ",client.indices.get_alias(index="*"))
        
    def test_add_document(self, es_client):
        doc = {"title": "Test Document", "content": "This is a test"}
        result = es_client.index(index="test_index", body=doc)
        assert result['result'] == 'created'
        es_client.indices.delete(index=['test_index'], ignore=[400, 404])

    def test_modify_document(self, es_client):
        # First, add a document
        doc = {"title": "Original Title", "content": "Original content"}
        result = es_client.index(index="test_index", id="1", body=doc)
        
        # Now, modify the document
        updated_doc = {"title": "Updated Title", "content": "Updated content"}
        result = es_client.index(index="test_index", id="1", body=updated_doc)
        assert result['result'] == 'updated'

    def test_query_document(self, es_client):
        # Add a document to query
        doc = {"title": "Queryable Document", "content": "This document can be queried"}
        es_client.index(index="test_index", id="2", body=doc)
        es_client.indices.refresh(index="test_index")

        # Perform a search
        result = es_client.search(index="test_index", body={"query": {"match": {"title": "Queryable"}}})
        assert result['hits']['total']['value'] == 1
        assert result['hits']['hits'][0]['_source']['title'] == "Queryable Document"

    def test_bulk_insert(self, es_client):
        docs = [
            {"_index": "test_index", "_source": {"title": f"Bulk Doc {i}", "content": f"Bulk content {i}"}}
            for i in range(5)
        ]
        success, _ = bulk(es_client, docs)
        assert success == 5
    
    def test_create_mapping(self, es_client):
        
        
        
        body = {"mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "content": {"type": "keyword"},
                }
            }
        }
        
        es_client.indices.create(index="test_index", body= body )
        
        doc = {"title": "Test Document", "content": "Non-Fiction"}
        es_client.index(index="test_index", id="3", body=doc)
        es_client.indices.refresh(index="test_index")
        
        result = es_client.search(index="test_index", body={"query": {"match": {"title": "Test Docume"}}})
        assert result['hits']['total']['value'] == 1
        
        result = es_client.search(index="test_index", body={"query": {"term": {"content": "Non-Fictio"}}})
        assert result['hits']['total']['value'] == 0
        
        result = es_client.search(index="test_index", body={"query": {"term": {"content": "Non-Fiction"}}})
        assert result['hits']['total']['value'] == 1
    
    def test_boolean_query(self, book_es_client):
        
        result = book_es_client.search(index="book_index", body={"query": 
                                                                    {"bool":
                                                                        {"must":
                                                                            {"match": {"title": "Book1"}}
                                                                        }
                                                                    }
                                                                })
        assert result['hits']['total']['value'] == 1

    def test_delete_document(self, es_client):
        # Add a document to delete
        doc = {"title": "Delete Me", "content": "This document will be deleted"}
        es_client.index(index="test_index", id="3", body=doc)

        # Delete the document
        result = es_client.delete(index="test_index", id="3")
        assert result['result'] == 'deleted'

        # Verify deletion
        with pytest.raises(Exception):
            es_client.get(index="test_index", id="3")
