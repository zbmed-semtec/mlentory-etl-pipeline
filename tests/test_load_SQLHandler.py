import pytest
import sys
from rdflib import Graph, URIRef, Literal
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from typing import Dict, Any


sys.path.append(".")
from load.core.dbHandler.SQLHandler import SQLHandler

class TestSQLHandler:
    @pytest.fixture
    def sql_handler(self):
        handler = SQLHandler(
            host="postgres",
            user="test_user",
            password="test_password",
            database="test_db"
        )
        return handler

    @pytest.fixture
    def mock_cursor(self):
        cursor = Mock()
        cursor.fetchone.return_value = [1]
        cursor.description = [("id",), ("name",)]
        cursor.fetchall.return_value = [(1, "test"), (2, "test2")]
        return cursor

    @pytest.fixture
    def mock_connection(self, mock_cursor):
        connection = Mock()
        connection.cursor.return_value = mock_cursor
        return connection
    
    @pytest.fixture(scope="function")
    def setup_and_teardown_sql_db(self, sql_handler):
        sql_handler.connect()
        cursor = sql_handler.connection.cursor()
        
        cursor.execute("DROP TABLE IF EXISTS test_triples")
        sql_handler.connection.commit()
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS "test_triples" (
                id BIGSERIAL PRIMARY KEY,
                subject VARCHAR(255),
                predicate VARCHAR(255),
                object VARCHAR(255)
            )
        """
        )
        sql_handler.connection.commit()

        yield

        
        cursor.close()
        sql_handler.connection.close()
        sql_handler.disconnect()

    def test_connect(self, sql_handler):
        with patch('psycopg2.connect') as mock_connect:
            sql_handler.connect()
            mock_connect.assert_called_once_with(
                host="postgres",
                user="test_user",
                password="test_password",
                database="test_db"
            )

    def test_disconnect(self, sql_handler):
        mock_conn = Mock()
        sql_handler.connection = mock_conn
        sql_handler.disconnect()
        mock_conn.close.assert_called_once()

    def test_insert(self, sql_handler, setup_and_teardown_sql_db):
        setup_and_teardown_sql_db
        
        result = sql_handler.query("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'test_triples'
        """)
        
        # Print the columns
        print("\nTest Triples Table Columns:")
        print(result)
        
        test_data = {
            "subject": "test_subject",
            "predicate": "test_predicate", 
            "object": "test_object"
        }
        result = sql_handler.insert("test_triples", test_data)
        assert result == 1

    
    def test_query(self, sql_handler, setup_and_teardown_sql_db):
        # First insert some test data
        test_data = {
            "subject": "test_subject",
            "predicate": "test_predicate",
            "object": "test_object"
        }
        sql_handler.insert("test_triples", test_data)
        
        # Then query it
        result = sql_handler.query("SELECT * FROM test_triples")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["subject"] == "test_subject"


    def test_delete(self, sql_handler, setup_and_teardown_sql_db):
        # First insert test data
        test_data = {
            "subject": "test_subject",
            "predicate": "test_predicate",
            "object": "test_object"
        }
        sql_handler.insert("test_triples", test_data)
        
        # Then delete it
        sql_handler.delete("test_triples", "subject = 'test_subject'")
        
        # Verify deletion
        result = sql_handler.query("SELECT * FROM test_triples")
        assert len(result) == 0

    def test_update(self, sql_handler, setup_and_teardown_sql_db):
        # First insert test data
        test_data = {
            "subject": "test_subject",
            "predicate": "test_predicate",
            "object": "test_object"
        }
        sql_handler.insert("test_triples", test_data)
        
        # Update the data
        update_data = {"object": "updated_object"}
        sql_handler.update("test_triples", update_data, "subject = 'test_subject'")
        
        # Verify update
        result = sql_handler.query("SELECT * FROM test_triples")
        assert result.iloc[0]["object"] == "updated_object"

    def test_execute_sql(self, sql_handler, mock_connection, mock_cursor):
        sql_handler.connection = mock_connection
        test_sql = "CREATE TABLE test_table (id INT)"
        
        sql_handler.execute_sql(test_sql)
        
        mock_cursor.execute.assert_called_once_with(test_sql)
        mock_connection.commit.assert_called_once()

    def test_reset_all_tables(self, sql_handler, mock_connection, mock_cursor):
        sql_handler.connection = mock_connection
        mock_cursor.fetchall.return_value = [("table1",), ("table2",)]
        
        sql_handler.reset_all_tables()
        
        assert mock_cursor.execute.call_count == 4
        mock_connection.commit.assert_called_once()