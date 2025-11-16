import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.pool import SimpleConnectionPool
import numpy as np
import pandas as pd
from typing import Callable, List, Dict, Set, Any, Optional

import time
import logging
import contextlib

register_adapter(np.int64, AsIs)
register_adapter(np.float64, AsIs)


class SQLHandler:
    """
    Handler for SQL database operations with connection pooling.

    This class provides functionality to:
    - Manage database connections with connection pooling
    - Execute CRUD operations
    - Handle batch operations
    - Clean and reset database state

    Attributes:
        host (str): Database host address
        user (str): Database username
        password (str): Database password
        database (str): Database name
        pool: Connection pool
        query_stats: Dictionary to store query execution statistics
        max_retries (int): Maximum number of reconnection attempts
        retry_delay (float): Delay between reconnection attempts in seconds
        min_connections (int): Minimum number of connections in pool
        max_connections (int): Maximum number of connections in pool
    """

    def __init__(
        self, 
        host: str, 
        user: str, 
        password: str, 
        database: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        min_connections: int = 1,
        max_connections: int = 10
    ):
        """
        Initialize SQLHandler with connection parameters and pooling.

        Args:
            host (str): Database host address
            user (str): Database username
            password (str): Database password
            database (str): Database name
            max_retries (int): Maximum number of reconnection attempts
            retry_delay (float): Delay between reconnection attempts in seconds
            min_connections (int): Minimum number of connections in pool
            max_connections (int): Maximum number of connections in pool

        Raises:
            ValueError: If any required connection parameter is empty
        """
        if not all([host, user, password, database]):
            raise ValueError("All connection parameters (host, user, password, database) must be provided")
            
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.pool = None
        self.query_stats = {"queries": {}}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.logger = logging.getLogger(__name__)

    def _create_pool(self) -> None:
        """
        Create a new connection pool with the configured parameters.
        
        Raises:
            psycopg2.Error: If unable to create the connection pool
        """
        try:
            self.pool = SimpleConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                # Configure TCP keepalive
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            self.logger.info(f"Created connection pool (min={self.min_connections}, max={self.max_connections})")
        except psycopg2.Error as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise

    @contextlib.contextmanager
    def get_connection(self):
        """
        Get a connection from the pool with automatic return.
        
        Yields:
            psycopg2.extensions.connection: Database connection from the pool
             
        Raises:
            psycopg2.Error: If unable to get a connection from the pool
        """
        if self.pool is None:
            self._create_pool()
             
        for attempt in range(self.max_retries + 1):
            try:
                connection = self.pool.getconn()
                try:
                    yield connection
                finally:
                    self.pool.putconn(connection)
                return
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                if attempt < self.max_retries:
                    self.logger.warning(
                        f"Failed to get connection from pool (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Recreating pool..."
                    )
                    try:
                        self.pool.closeall()
                    except:
                        pass
                    self._create_pool()
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed to get connection after {self.max_retries + 1} attempts")
                    raise

    def connect(self) -> None:
        """
        Initialize the connection pool.
        
        Raises:
            psycopg2.Error: If unable to create the connection pool
        """
        self._create_pool()

    def disconnect(self) -> None:
        """Close all connections in the pool."""
        if self.pool:
            try:
                self.pool.closeall()
                self.logger.info("Closed all database connections in the pool")
            except:
                pass
            finally:
                self.pool = None

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection pool.

        Returns:
            Dict[str, Any]: Health status information including pool status,
                           query statistics, and connection parameters

        Example:
            {
                "status": "healthy",
                "pool_active": True,
                "host": "postgres_db",
                "database": "history_DB",
                "query_count": 42,
                "total_queries": 15,
                "min_connections": 1,
                "max_connections": 10
            }
        """
        pool_active = False
        if self.pool:
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                        pool_active = True
            except:
                pass
                
        return {
            "status": "healthy" if pool_active else "unhealthy",
            "pool_active": pool_active,
            "host": self.host,
            "database": self.database,
            "query_count": sum(stats["count"] for stats in self.query_stats["queries"].values()),
            "total_queries": len(self.query_stats["queries"]),
            "min_connections": self.min_connections,
            "max_connections": self.max_connections
        }

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """
        Insert a new record into the specified table.

        Args:
            table (str): Target table name
            data (Dict[str, Any]): Column names and values to insert

        Returns:
            int: ID of the inserted row
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
            ValueError: If table name or data is invalid
        """
        if not table or not data:
            raise ValueError("Table name and data must be provided")
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                placeholders = ", ".join(["%s"] * len(data))
                columns = ", ".join(f'"{k}"' for k in data.keys())
                sql = f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders}) RETURNING id'
                cursor.execute(sql, list(data.values()))
                last_insert_id = cursor.fetchone()[0]
                conn.commit()
                return last_insert_id
            finally:
                cursor.close()

    def query(self, sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.

        Args:
            sql (str): SQL query to execute
            params (Dict[str, Any], optional): Query parameters

        Returns:
            pd.DataFrame: Query results as DataFrame
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
            ValueError: If SQL query is empty
        """
        if not sql or not sql.strip():
            raise ValueError("SQL query must be provided")
            
        start_time = time.time()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params or ())
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                result = cursor.fetchall()
                
                duration = time.time() - start_time

                if sql not in self.query_stats["queries"]:
                    self.query_stats["queries"][sql] = {"count": 0, "total_time": 0}

                self.query_stats["queries"][sql]["count"] += 1
                self.query_stats["queries"][sql]["total_time"] += duration

                return pd.DataFrame(result, columns=columns)
            finally:
                cursor.close()

    def delete(self, table: str, condition: str) -> None:
        """
        Delete records from a table based on condition.

        Args:
            table (str): Target table name
            condition (str): WHERE clause for deletion
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
            ValueError: If table name or condition is invalid
        """
        if not table or not condition:
            raise ValueError("Table name and condition must be provided")
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                sql = f'DELETE FROM "{table}" WHERE {condition}'
                cursor.execute(sql)
                conn.commit()
            finally:
                cursor.close()

    def update(self, table: str, data: Dict[str, Any], condition: str) -> None:
        """
        Update records in a table based on condition.

        Args:
            table (str): Target table name
            data (Dict[str, Any]): New values to set
            condition (str): WHERE clause for update
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
            ValueError: If parameters are invalid
        """
        if not table or not data or not condition:
            raise ValueError("Table name, data, and condition must be provided")
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                set_clause = ", ".join([f'"{key}" = %s' for key in data.keys()])
                sql = f'UPDATE "{table}" SET {set_clause} WHERE {condition}'
                cursor.execute(sql, list(data.values()))
                conn.commit()
            finally:
                cursor.close()

    def execute_sql(self, sql: str, params: tuple = None) -> None:
        """
        Execute a SQL query without returning results.

        Args:
            sql (str): SQL query to execute
            params (tuple, optional): Query parameters

        Raises:
            psycopg2.Error: If there's an error executing the SQL query
            ValueError: If SQL query is empty
        """
        if not sql or not sql.strip():
            raise ValueError("SQL query must be provided")
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params or ())
                conn.commit()
            finally:
                cursor.close()

    def delete_all_tables(self):
        """
        Delete all tables in the database.
        
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
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

                conn.commit()
            finally:
                cursor.close()

    def clean_all_tables(self):
        """
        Remove all data from tables while preserving structure and reset sequences.

        This method:
        1. Defers all constraints
        2. Deletes all data from tables
        3. Resets all sequences (auto-increment counters) to 1
        
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
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

                conn.commit()
            finally:
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
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
            ValueError: If parameters are invalid
        """
        if not table or not columns:
            raise ValueError("Table name and columns must be provided")
            
        if not values:
            return []

        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                inserted_ids = []
                columns_str = ", ".join(f'"{col}"' for col in columns)

                # Construct the INSERT query for use with execute_values
                query = f'INSERT INTO "{table}" ({columns_str}) VALUES %s RETURNING id'

                # Process in batches
                for i in range(0, len(values), batch_size):
                    batch = values[i : i + batch_size]
                    for row in batch:
                        row = tuple(self.clean_extracted_text(value) if isinstance(value, (bytes, str)) else value for value in row)
                    
                    # Use execute_values with fetch=True to get the returned IDs
                    batch_ids = execute_values(cursor, query, batch, fetch=True)
                    
                    inserted_ids.extend([row[0] for row in batch_ids])

                conn.commit()
                return inserted_ids
            finally:
                cursor.close()

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
            batch_size (int): The size of each batch to process

        Raises:
            psycopg2.Error: If there's an error executing the SQL query
            ValueError: If parameters are invalid
        """
        if not table or not updates or not conditions:
            raise ValueError("Table name, updates, and conditions must be provided")
            
        if len(updates) != len(conditions):
            raise ValueError("Updates and conditions lists must have the same length")
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                for i in range(0, len(updates), batch_size):
                    batch_updates = updates[i : min(i + batch_size, len(updates))]
                    batch_conditions = conditions[i : min(i + batch_size, len(conditions))]

                    for update_dict, condition in zip(batch_updates, batch_conditions):
                        set_clause = ", ".join([f'"{key}" = %s' for key in update_dict.keys()])
                        sql = f'UPDATE "{table}" SET {set_clause} WHERE {condition}'
                        
                        cursor.execute(sql, list(update_dict.values()))

                conn.commit()
            finally:
                cursor.close()

    def batch_query(self, queries: List[str]) -> List[pd.DataFrame]:
        """
        Execute multiple queries in batch and return results.

        Args:
            queries (List[str]): List of SQL queries to execute

        Returns:
            List[pd.DataFrame]: List of query results as DataFrames
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
            ValueError: If queries list is invalid
        """
        if not queries:
            raise ValueError("Queries list must be provided")
            
        results = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                for query in queries:
                    if not query or not query.strip():
                        results.append(pd.DataFrame())
                        continue
                        
                    cursor.execute(query)
                    if cursor.description:  # If the query returns results
                        columns = [desc[0] for desc in cursor.description]
                        result = cursor.fetchall()
                        results.append(pd.DataFrame(result, columns=columns))
                    else:
                        results.append(pd.DataFrame())
            finally:
                cursor.close()
                
        return results

    def clean_extracted_text(self, extracted_text: str | bytes) -> str:
        if isinstance(extracted_text, bytes):
            content = extracted_text.decode("utf-8", errors="replace").replace(
                "\x00", "\ufffd"
            )
        else:
            content = extracted_text.replace("\x00", "\ufffd")

        return content