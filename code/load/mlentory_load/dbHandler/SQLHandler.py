import psycopg2
import pandas as pd
from typing import Callable, List, Dict, Set, Any
from psycopg2.extras import execute_values
from psycopg2.extensions import register_adapter, AsIs
import numpy as np

register_adapter(np.int64, AsIs)
register_adapter(np.float64, AsIs)

import time


class SQLHandler:
    """
    Handler for SQL database operations.

    This class provides functionality to:
    - Manage database connections
    - Execute CRUD operations
    - Handle batch operations
    - Clean and reset database state

    Attributes:
        host (str): Database host address
        user (str): Database username
        password (str): Database password
        database (str): Database name
        connection: Active database connection
        query_stats: Dictionary to store query execution statistics
    """

    def __init__(self, host: str, user: str, password: str, database: str):
        """
        Initialize SQLHandler with connection parameters.

        Args:
            host (str): Database host address
            user (str): Database username
            password (str): Database password
            database (str): Database name
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.query_stats = {"queries": {}}

    def connect(self) -> None:
        """Establish connection to the PostgreSQL database."""
        self.connection = psycopg2.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
        )

    def disconnect(self) -> None:
        """Close the active database connection if it exists."""
        if self.connection:
            self.connection.close()

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """
        Insert a new record into the specified table.

        Args:
            table (str): Target table name
            data (Dict[str, Any]): Column names and values to insert

        Returns:
            int: ID of the inserted row
        """
        cursor = self.connection.cursor()
        placeholders = ", ".join(["%s"] * len(data))
        columns = ", ".join(f'"{k}"' for k in data.keys())
        sql = f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders}) RETURNING id'
        cursor.execute(sql, list(data.values()))
        last_insert_id = cursor.fetchone()[0]
        self.connection.commit()
        cursor.close()
        return last_insert_id

    def query(self, sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.

        Args:
            sql (str): SQL query to execute
            params (Dict[str, Any], optional): Query parameters

        Returns:
            pd.DataFrame: Query results as DataFrame
        """
        start_time = time.time()
        cursor = self.connection.cursor()
        cursor.execute(sql, params or ())
        columns = [desc[0] for desc in cursor.description]
        result = cursor.fetchall()
        cursor.close()
        duration = time.time() - start_time

        if sql not in self.query_stats["queries"]:
            self.query_stats["queries"][sql] = {"count": 0, "total_time": 0}

        self.query_stats["queries"][sql]["count"] += 1
        self.query_stats["queries"][sql]["total_time"] += duration

        return pd.DataFrame(result, columns=columns)

    def delete(self, table: str, condition: str) -> None:
        """
        Delete records from a table based on condition.

        Args:
            table (str): Target table name
            condition (str): WHERE clause for deletion
        """
        cursor = self.connection.cursor()
        sql = f'DELETE FROM "{table}" WHERE {condition}'
        cursor.execute(sql)
        self.connection.commit()
        cursor.close()

    def update(self, table: str, data: Dict[str, Any], condition: str) -> None:
        """
        Update records in a table based on condition.

        Args:
            table (str): Target table name
            data (Dict[str, Any]): New values to set
            condition (str): WHERE clause for update
        """
        cursor = self.connection.cursor()
        set_clause = ", ".join([f'"{key}" = %s' for key in data.keys()])
        sql = f'UPDATE "{table}" SET {set_clause} WHERE {condition}'
        cursor.execute(sql, list(data.values()))
        self.connection.commit()
        cursor.close()

    def execute_sql(self, sql: str, params: tuple = None) -> None:
        """
        Execute a SQL query without returning results.

        Args:
            sql (str): SQL query to execute
            params (tuple, optional): Query parameters

        Raises:
            psycopg2.Error: If there's an error executing the SQL query
        """
        cursor = self.connection.cursor()
        cursor.execute(sql, params or ())
        self.connection.commit()
        cursor.close()

    def delete_all_tables(self):
        """Delete all tables in the database."""

        cursor = self.connection.cursor()
        cursor.execute(
            """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """
        )
        tables = cursor.fetchall()

        cursor.execute("SET CONSTRAINTS ALL DEFERRED")
        for table in tables:
            cursor.execute(f'TRUNCATE TABLE "{table[0]}" CASCADE')

        self.connection.commit()

        cursor.close()

    def clean_all_tables(self):
        """
        Remove all data from tables while preserving structure and reset sequences.

        This method:
        1. Defers all constraints
        2. Deletes all data from tables
        3. Resets all sequences (auto-increment counters) to 1
        """
        cursor = self.connection.cursor()

        # Get all tables
        cursor.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        )
        tables = cursor.fetchall()

        # Defer constraints and truncate tables
        cursor.execute("SET CONSTRAINTS ALL DEFERRED")

        for table in tables:
            table_name = table[0]
            # TRUNCATE is faster than DELETE and resets sequences automatically
            cursor.execute(f'TRUNCATE TABLE "{table_name}" CASCADE')

        self.connection.commit()
        cursor.close()

    def batch_insert(
        self,
        table: str,
        columns: List[str],
        values: List[tuple],
        batch_size: int = 1000,
    ) -> List[int]:
        """
        Insert multiple records into the specified table in batches using
        psycopg2.extras.execute_values for efficiency and type safety.

        Args:
            table (str): Target table name
            columns (List[str]): List of column names
            values (List[tuple]): List of value tuples to insert
            batch_size (int): The size of each batch to process.

        Returns:
            List[int]: List of inserted row IDs
        """
        if not values:
            return []

        cursor = self.connection.cursor()
        inserted_ids = []
        columns_str = ", ".join(f'"{col}"' for col in columns)

        # Construct the INSERT query for use with execute_values
        query = f'INSERT INTO "{table}" ({columns_str}) VALUES %s RETURNING id'

        # Process in batches
        for i in range(0, len(values), batch_size):
            batch = values[i : i + batch_size]
            
            # Use execute_values with fetch=True to get the returned IDs
            batch_ids = execute_values(cursor, query, batch, fetch=True)
            
            inserted_ids.extend([row[0] for row in batch_ids])

        self.connection.commit()
        cursor.close()
        return inserted_ids

    def batch_update(
        self,
        table: str,
        updates: List[Dict[str, Any]],
        conditions: List[str],
        batch_size: int = 1000,
    ) -> None:
        """
        Update multiple records in a table based on conditions.

        Args:
            table (str): Target table name
            updates (List[Dict[str, Any]]): List of update dictionaries
            conditions (List[str]): List of WHERE clauses for updates
        """
        cursor = self.connection.cursor()

        for i in range(0, len(updates), batch_size):
            batch_updates = updates[i : min(i + batch_size, len(updates))]
            batch_conditions = conditions[i : min(i + batch_size, len(conditions))]

            for update_dict, condition in zip(batch_updates, batch_conditions):
                set_clause = ", ".join([f'"{key}" = %s' for key in update_dict.keys()])
                sql = f'UPDATE "{table}" SET {set_clause} WHERE {condition}'
                
                execute_values(cursor, sql, list(update_dict.values()))
                

        self.connection.commit()
        cursor.close()

    def batch_query(self, queries: List[str]) -> List[pd.DataFrame]:
        """
        Execute multiple queries in batch and return results.

        Args:
            queries (List[str]): List of SQL queries to execute

        Returns:
            List[pd.DataFrame]: List of query results as DataFrames
        """
        results = []
        cursor = self.connection.cursor()

        for query in queries:
            cursor.execute(query)
            if cursor.description:  # If the query returns results
                columns = [desc[0] for desc in cursor.description]
                result = cursor.fetchall()
                results.append(pd.DataFrame(result, columns=columns))
            else:
                results.append(pd.DataFrame())

        cursor.close()
        return results
