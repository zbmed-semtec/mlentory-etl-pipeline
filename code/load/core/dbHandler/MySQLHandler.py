import mysql.connector
import pandas as pd
from typing import Callable, List, Dict,Set

class MySQLHandler:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def disconnect(self):
        if self.connection:
            self.connection.close()

    # db.insert('table_name', {'column1': 'value1', 'column2': 'value2'})
    def insert(self, table, data):
        cursor = self.connection.cursor()
        placeholders = ', '.join(['%s'] * len(data))
        columns = ', '.join(data.keys())
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, list(data.values()))
        self.connection.commit()
        cursor.close()

    def query(self, sql:str, params:Dict = None) -> pd.DataFrame:
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(sql, params or ())
        result = cursor.fetchall()
        cursor.close()
        return pd.DataFrame(result)
    
    def delete(self, table, condition):
        cursor = self.connection.cursor()
        sql = f"DELETE FROM {table} WHERE {condition}"
        cursor.execute(sql)
        self.connection.commit()
        cursor.close()

    def update(self, table, data, condition):
        cursor = self.connection.cursor()
        set_clause = ', '.join([f"{key} = %s" for key in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        cursor.execute(sql, list(data.values()))
        self.connection.commit()
        cursor.close()