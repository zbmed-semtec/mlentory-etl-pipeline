import psycopg2
from psycopg2 import pool
import pandas as pd
from typing import Callable, List, Dict, Set, Any, Optional
from psycopg2.extras import execute_values
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import time
import logging

register_adapter(np.int64, AsIs)
register_adapter(np.float64, AsIs)


class SQLHandler:
    """
    Handler for SQL database operations with connection pooling and health checking.

    This class provides functionality to:
    - Manage database connections with automatic reconnection
    - Execute CRUD operations with connection pooling
    - Handle batch operations
    - Clean and reset database state

    Attributes:
        host (str): Database host address
        user (str): Database username
        password (str): Database password
        database (str): Database name
        connection_pool: PostgreSQL connection pool
        query_stats: Dictionary to store query execution statistics
        min_connections (int): Minimum number of connections in pool
        max_connections (int): Maximum number of connections in pool
    """

    def __init__(
        self, 
        host: str, 
        user: str, 
        password: str, 
        database: str,
        min_connections: int = 2,
        max_connections: int = 10
    ):
        """
        Initialize SQLHandler with connection parameters and pooling.

        Args:
            host (str): Database host address
            user (str): Database username
            password (str): Database password
            database (str): Database name
            min_connections (int): Minimum connections in pool (default: 2)
            max_connections (int): Maximum connections in pool (default: 10)
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_pool: Optional[pool.SimpleConnectionPool] = None
        self.query_stats = {"queries": {}}
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """
        Establish connection pool to the PostgreSQL database.
        
        Raises:
            psycopg2.Error: If unable to create connection pool
        """
        try:
            if self.connection_pool:
                self.connection_pool.closeall()
            
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                self.min_connections,
                self.max_connections,
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                # Connection health parameters
                keepalives_idle=600,      # Start keepalives after 10 minutes idle
                keepalives_interval=30,   # Send keepalive every 30 seconds
                keepalives_count=3,       # Allow 3 failed keepalives before declaring dead
                connect_timeout=10        # 10 second connection timeout
            )
            self.logger.info(f"Connection pool created with {self.min_connections}-{self.max_connections} connections")
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise

    def disconnect(self) -> None:
        """Close all connections in the pool if it exists."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None
            self.logger.info("Connection pool closed")

    def _get_connection(self):
        """
        Get a connection from the pool with automatic reconnection.
        
        Returns:
            psycopg2.connection: Database connection
            
        Raises:
            psycopg2.Error: If unable to get connection after retries
        """
        if not self.connection_pool:
            self.logger.warning("Connection pool not initialized, creating new pool")
            self.connect()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                connection = self.connection_pool.getconn()
                
                # Test the connection with a simple query
                if connection.closed != 0:
                    # Connection is closed, remove it and try again
                    self.connection_pool.putconn(connection, close=True)
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise psycopg2.InterfaceError("Unable to get valid connection")
                
                return connection
                
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                self.logger.warning(f"Connection issue on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    # Try to recreate the pool
                    try:
                        self.connect()
                    except Exception as pool_error:
                        self.logger.error(f"Failed to recreate pool: {pool_error}")
                        time.sleep(1)  # Brief delay before retry
                else:
                    raise

    def _return_connection(self, connection, close: bool = False) -> None:
        """
        Return a connection to the pool.
        
        Args:
            connection: Database connection to return
            close (bool): Whether to close the connection instead of returning it
        """
        if self.connection_pool and connection:
            self.connection_pool.putconn(connection, close=close)

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
        """
        connection = self._get_connection()
        try:
            cursor = connection.cursor()
            placeholders = ", ".join(["%s"] * len(data))
            columns = ", ".join(f'"{k}"' for k in data.keys())
            sql = f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders}) RETURNING id'
            cursor.execute(sql, list(data.values()))
            last_insert_id = cursor.fetchone()[0]
            connection.commit()
            cursor.close()
            return last_insert_id
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Insert failed: {e}")
            raise
        finally:
            self._return_connection(connection)

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
        """
        start_time = time.time()
        connection = self._get_connection()
        
        try:
            cursor = connection.cursor()
            cursor.execute(sql, params or ())
            columns = [desc[0] for desc in cursor.description]
            result = cursor.fetchall()
            cursor.close()
            
            duration = time.time() - start_time
            
            # Update query statistics
            if sql not in self.query_stats["queries"]:
                self.query_stats["queries"][sql] = {"count": 0, "total_time": 0}
            
            self.query_stats["queries"][sql]["count"] += 1
            self.query_stats["queries"][sql]["total_time"] += duration
            
            return pd.DataFrame(result, columns=columns)
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise
        finally:
            self._return_connection(connection)

    def delete(self, table: str, condition: str) -> None:
        """
        Delete records from a table based on condition.

        Args:
            table (str): Target table name
            condition (str): WHERE clause for deletion
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
        """
        connection = self._get_connection()
        try:
            cursor = connection.cursor()
            sql = f'DELETE FROM "{table}" WHERE {condition}'
            cursor.execute(sql)
            connection.commit()
            cursor.close()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Delete failed: {e}")
            raise
        finally:
            self._return_connection(connection)

    def update(self, table: str, data: Dict[str, Any], condition: str) -> None:
        """
        Update records in a table based on condition.

        Args:
            table (str): Target table name
            data (Dict[str, Any]): New values to set
            condition (str): WHERE clause for update
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
        """
        connection = self._get_connection()
        try:
            cursor = connection.cursor()
            set_clause = ", ".join([f'"{key}" = %s' for key in data.keys()])
            sql = f'UPDATE "{table}" SET {set_clause} WHERE {condition}'
            cursor.execute(sql, list(data.values()))
            connection.commit()
            cursor.close()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Update failed: {e}")
            raise
        finally:
            self._return_connection(connection)

    def execute_sql(self, sql: str, params: tuple = None) -> None:
        """
        Execute a SQL query without returning results.

        Args:
            sql (str): SQL query to execute
            params (tuple, optional): Query parameters

        Raises:
            psycopg2.Error: If there's an error executing the SQL query
        """
        connection = self._get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(sql, params or ())
            connection.commit()
            cursor.close()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Execute SQL failed: {e}")
            raise
        finally:
            self._return_connection(connection)

    def delete_all_tables(self):
        """
        Delete all tables in the database.
        
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
        """
        connection = self._get_connection()
        try:
            cursor = connection.cursor()
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

            connection.commit()
            cursor.close()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Delete all tables failed: {e}")
            raise
        finally:
            self._return_connection(connection)

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
        connection = self._get_connection()
        try:
            cursor = connection.cursor()

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

            connection.commit()
            cursor.close()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Clean all tables failed: {e}")
            raise
        finally:
            self._return_connection(connection)

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
        """
        if not values:
            return []

        connection = self._get_connection()
        try:
            cursor = connection.cursor()
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

            connection.commit()
            cursor.close()
            return inserted_ids
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Batch insert failed: {e}")
            raise
        finally:
            self._return_connection(connection)

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
            batch_size (int): The size of each batch to process.
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
        """
        connection = self._get_connection()
        try:
            cursor = connection.cursor()

            for i in range(0, len(updates), batch_size):
                batch_updates = updates[i : min(i + batch_size, len(updates))]
                batch_conditions = conditions[i : min(i + batch_size, len(conditions))]

                for update_dict, condition in zip(batch_updates, batch_conditions):
                    set_clause = ", ".join([f'"{key}" = %s' for key in update_dict.keys()])
                    sql = f'UPDATE "{table}" SET {set_clause} WHERE {condition}'
                    
                    cursor.execute(sql, list(update_dict.values()))

            connection.commit()
            cursor.close()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Batch update failed: {e}")
            raise
        finally:
            self._return_connection(connection)

    def batch_query(self, queries: List[str]) -> List[pd.DataFrame]:
        """
        Execute multiple queries in batch and return results.

        Args:
            queries (List[str]): List of SQL queries to execute

        Returns:
            List[pd.DataFrame]: List of query results as DataFrames
            
        Raises:
            psycopg2.Error: If there's an error executing the SQL query
        """
        connection = self._get_connection()
        try:
            results = []
            cursor = connection.cursor()

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
        except Exception as e:
            self.logger.error(f"Batch query failed: {e}")
            raise
        finally:
            self._return_connection(connection)

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get current connection pool status for monitoring.
        
        Returns:
            Dict[str, Any]: Pool status information
        """
        if not self.connection_pool:
            return {"status": "not_initialized"}
        
        # Note: SimpleConnectionPool doesn't expose detailed stats
        # This is a basic implementation
        return {
            "status": "active",
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
            "pool_type": "SimpleConnectionPool"
        }
