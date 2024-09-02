import mysql.connector
import pandas as pd
from typing import Callable, List, Dict, Set, Any

class MySQLHandler:
    def __init__(self, host: str, user: str, password: str, database: str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self) -> None:
        """
        Establishes a connection to the MySQL database.
        """
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def disconnect(self) -> None:
        """
        Closes the connection to the MySQL database if it exists.
        """
        if self.connection:
            self.connection.close()

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """
        Inserts a new record into the specified table.

        Args:
            table (str): The name of the table to insert into.
            data (Dict[str, Any]): A dictionary containing column names and values to insert.

        Returns:
            int: The ID of the last inserted row.
        """
        cursor = self.connection.cursor()
        placeholders = ', '.join(['%s'] * len(data))
        columns = ', '.join(data.keys())
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, list(data.values()))
        last_insert_id = cursor.lastrowid
        cursor.close()
        return last_insert_id


    def query(self, sql: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a pandas DataFrame.

        Args:
            sql (str): The SQL query to execute.
            params (Dict[str, Any], optional): A dictionary of parameters to be used in the query.

        Returns:
            pd.DataFrame: A DataFrame containing the query results.
        """
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(sql, params or ())
        result = cursor.fetchall()
        cursor.close()
        return pd.DataFrame(result)
    

    def delete(self, table: str, condition: str) -> None:
        """
        Deletes records from the specified table based on the given condition.

        Args:
            table (str): The name of the table to delete from.
            condition (str): The WHERE clause specifying which records to delete.
        """
        cursor = self.connection.cursor()
        sql = f"DELETE FROM {table} WHERE {condition}"
        cursor.execute(sql)
        self.connection.commit()
        cursor.close()


    def update(self, table: str, data: Dict[str, Any], condition: str) -> None:
        """
        Updates records in the specified table based on the given condition.

        Args:
            table (str): The name of the table to update.
            data (Dict[str, Any]): A dictionary containing column names and new values to update.
            condition (str): The WHERE clause specifying which records to update.
        """
        cursor = self.connection.cursor()
        set_clause = ', '.join([f"{key} = %s" for key in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        cursor.execute(sql, list(data.values()))
        self.connection.commit()
        cursor.close()
    
    def execute_sql(self, sql: str) -> None:
        """
        Executes a SQL query without returning any results.
        
        Args:
            sql (str): The SQL query to execute.
        """
        cursor = self.connection.cursor()
        cursor.execute(sql)
        self.connection.commit()
        cursor.close()
        
    
    def reset_all_tables(self):
        cursor = self.connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        for table in tables:
            cursor.execute(f"TRUNCATE TABLE {table[0]}")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        self.connection.commit()
        cursor.close()