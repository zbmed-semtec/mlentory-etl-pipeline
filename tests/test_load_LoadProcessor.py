import pytest
import sys
from rdflib import Graph, URIRef, Literal


sys.path.append('.')
from load.core.LoadProcessor import LoadProcessor

class TestLoadProcessor:

    @pytest.fixture(scope="class")
    def load_processor(self):
        return LoadProcessor(host='mysql', user='test_user', password='test_pass', database='test_db', port=33061)

    @pytest.fixture(scope="function")
    def setup_and_teardown_sql_db(self, load_processor):
        load_processor.connect_to_mysql()
        cursor = load_processor.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_triples (
                subject VARCHAR(255),
                predicate VARCHAR(255),
                object VARCHAR(255)
            )
        """)
        load_processor.connection.commit()
        
        yield
        
        cursor.execute("DROP TABLE IF EXISTS test_triples")
        load_processor.connection.commit()
        cursor.close()
        load_processor.connection.close()
    
    def test_connect_to_mysql_mock(self, load_processor, mocker):
        mock_connect = mocker.patch('mysql.connector.connect')
        load_processor.connect_to_mysql()
        mock_connect.assert_called_once_with(
            host='mysql',
            user='test_user',
            password='test_pass',
            database='test_db'
        )

    def test_load_graph_to_mysql_mock(self, load_processor, mocker):
        mock_cursor = mocker.MagicMock()
        mock_connection = mocker.MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        load_processor.connection = mock_connection

        test_graph = Graph()
        test_graph.add((URIRef('http://example.org/subject'), URIRef('http://example.org/predicate'), Literal('object')))

        load_processor.load_graph_to_mysql(test_graph)

        mock_cursor.execute.assert_called_once_with(
            "INSERT INTO triples (subject, predicate, object) VALUES (%s, %s, %s)",
            ('http://example.org/subject', 'http://example.org/predicate', 'object')
        )
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()

    def test_load_graph_to_mysql_real(self, load_processor, setup_and_teardown_sql_db):
        test_graph = Graph()
        test_graph.add((URIRef('http://example.org/subject'), URIRef('http://example.org/predicate'), Literal('object')))

        load_processor.load_graph_to_mysql(test_graph, table_name="test_triples")

        cursor = load_processor.connection.cursor()
        cursor.execute("SELECT * FROM test_triples")
        result = cursor.fetchall()

        assert len(result) == 1
        assert result[0] == ('http://example.org/subject', 'http://example.org/predicate', 'object')

        cursor.close()