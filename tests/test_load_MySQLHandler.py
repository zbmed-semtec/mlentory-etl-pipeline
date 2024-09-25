import pytest
import sys
from rdflib import Graph, URIRef, Literal


sys.path.append(".")
from load.core.dbHandler.MySQLHandler import MySQLHandler


class TestLMySQLHandler:
    @pytest.fixture(scope="class")
    def my_sql_handler(self):
        return MySQLHandler(
            host="mysql", user="test_user", password="test_pass", database="test_DB"
        )

    @pytest.fixture(scope="function")
    def setup_and_teardown_sql_db(self, my_sql_handler):
        my_sql_handler.connect()
        cursor = my_sql_handler.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_triples (
                subject VARCHAR(255),
                predicate VARCHAR(255),
                object VARCHAR(255)
            )
        """
        )
        my_sql_handler.connection.commit()

        yield

        cursor.execute("DROP TABLE IF EXISTS test_triples")
        my_sql_handler.connection.commit()
        cursor.close()
        my_sql_handler.connection.close()
        my_sql_handler.disconnect()

    def test_connect_to_mysql_mock(self, my_sql_handler, mocker):
        mock_connect = mocker.patch("mysql.connector.connect")
        my_sql_handler.connect()
        mock_connect.assert_called_once_with(
            host="mysql", user="test_user", password="test_pass", database="test_DB"
        )

    def test_load_graph_to_mysql_mock(self, my_sql_handler, mocker):
        mock_cursor = mocker.MagicMock()
        mock_connection = mocker.MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        my_sql_handler.connection = mock_connection

        my_sql_handler.insert(
            table="test_triples",
            data={
                "subject": "http://example.org/subject",
                "predicate": "http://example.org/predicate",
                "object": "object",
            },
        )

        mock_cursor.execute.assert_called_once_with(
            "INSERT INTO test_triples (subject, predicate, object) VALUES (%s, %s, %s)",
            ["http://example.org/subject", "http://example.org/predicate", "object"],
        )
        # mock_connection.commit.assert_called_once()
        # mock_cursor.close.assert_called_once()

    def test_load_graph_to_mysql_real(self, my_sql_handler, setup_and_teardown_sql_db):
        my_sql_handler.insert(
            table="test_triples",
            data={
                "subject": "http://example.org/subject",
                "predicate": "http://example.org/predicate",
                "object": "object",
            },
        )

        cursor = my_sql_handler.connection.cursor()
        cursor.execute("SELECT * FROM test_triples")
        result = cursor.fetchall()

        assert len(result) == 1
        assert result[0] == (
            "http://example.org/subject",
            "http://example.org/predicate",
            "object",
        )

        cursor.close()
