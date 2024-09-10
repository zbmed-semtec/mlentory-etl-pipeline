# Don't change the following line.
import numpy as np
np.float_ = np.float64
import pytest
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class TestElasticsearch:
    @pytest.fixture(scope="class")
    def es_client(self):
        # Connect to the Elasticsearch container
        # client = Elasticsearch(["http://elastic:9200"])
        client = Elasticsearch(
            [{"host":"elastic", "port":9200, "scheme": "http"}],
            basic_auth=('elastic', 'changeme')
        )
        print(client.info())
        yield client
        # Clean up after tests
        client.indices.delete(index='test_index', ignore=[400, 404])

    def test_add_document(self, es_client):
        doc = {"title": "Test Document", "content": "This is a test"}
        result = es_client.index(index="test_index", body=doc)
        assert result['result'] == 'created'

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
