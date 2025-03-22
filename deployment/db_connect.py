#!/usr/bin/env python3
"""
PostgreSQL Database Connection Utility

This script generates connection strings and information for connecting to the
main PostgreSQL database in the MLentory ETL pipeline.

Example:
    $ python db_connect.py
"""

import os
import sys
from typing import Dict, Any, Optional
import argparse
import urllib.parse


def get_postgres_connection_info() -> Dict[str, Any]:
    """
    Get connection information for the main PostgreSQL database.

    Returns:
        Dict[str, Any]: Dictionary containing connection parameters.

    Example:
        >>> info = get_postgres_connection_info()
        >>> print(info["jdbc_url"])
        jdbc:postgresql://localhost:5432/history_DB
    """
    # Connection parameters from docker-compose.yml
    connection_info = {
        "host": "localhost",
        "port": 5432,
        "database": "history_DB",
        "user": "user",
        "password": "password",
    }
    
    # Generate connection strings for different clients
    connection_info["jdbc_url"] = (
        f"jdbc:postgresql://{connection_info['host']}:{connection_info['port']}/"
        f"{connection_info['database']}"
    )
    
    connection_info["psql_command"] = (
        f"psql -h {connection_info['host']} -p {connection_info['port']} "
        f"-U {connection_info['user']} -d {connection_info['database']}"
    )
    
    # URL-encode password for connection strings that need it
    encoded_password = urllib.parse.quote_plus(connection_info["password"])
    
    connection_info["sqlalchemy_uri"] = (
        f"postgresql://{connection_info['user']}:{encoded_password}@"
        f"{connection_info['host']}:{connection_info['port']}/"
        f"{connection_info['database']}"
    )
    
    connection_info["python_connection_code"] = f"""
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="{connection_info['host']}",
    port={connection_info['port']},
    database="{connection_info['database']}",
    user="{connection_info['user']}",
    password="{connection_info['password']}"
)

# Create a cursor
cur = conn.cursor()

# Execute a query
cur.execute("SELECT version();")

# Fetch results
version = cur.fetchone()
print(version)

# Close cursor and connection
cur.close()
conn.close()
"""
    
    return connection_info


def print_connection_info(connection_info: Dict[str, Any], format_type: Optional[str] = None) -> None:
    """
    Print connection information in the specified format.

    Args:
        connection_info: Dictionary containing connection parameters.
        format_type: Type of connection string to print (psql, jdbc, sqlalchemy, python, or all).

    Raises:
        ValueError: If an invalid format type is provided.

    Example:
        >>> info = get_postgres_connection_info()
        >>> print_connection_info(info, "psql")
        PostgreSQL Command Line:
        psql -h localhost -p 5432 -U user -d history_DB
    """
    if format_type == "psql" or format_type is None:
        print("PostgreSQL Command Line:")
        print(connection_info["psql_command"])
        print(f"Password: {connection_info['password']}")
        print()
    
    if format_type == "jdbc" or format_type is None:
        print("JDBC Connection URL:")
        print(connection_info["jdbc_url"])
        print(f"Username: {connection_info['user']}")
        print(f"Password: {connection_info['password']}")
        print()
    
    if format_type == "sqlalchemy" or format_type is None:
        print("SQLAlchemy Connection URI:")
        print(connection_info["sqlalchemy_uri"])
        print()
    
    if format_type == "python" or format_type is None:
        print("Python Connection Code:")
        print(connection_info["python_connection_code"])
        print()
    
    if format_type not in [None, "psql", "jdbc", "sqlalchemy", "python"]:
        raise ValueError(f"Invalid format type: {format_type}")


def main() -> None:
    """
    Main function to parse arguments and display connection information.

    Raises:
        SystemExit: If an invalid format type is provided.

    Example:
        $ python db_connect.py --format psql
    """
    parser = argparse.ArgumentParser(
        description="Generate connection strings for PostgreSQL database"
    )
    parser.add_argument(
        "--format", 
        choices=["psql", "jdbc", "sqlalchemy", "python", "all"],
        default="all",
        help="Type of connection string to generate"
    )
    
    args = parser.parse_args()
    
    try:
        connection_info = get_postgres_connection_info()
        format_type = None if args.format == "all" else args.format
        print_connection_info(connection_info, format_type)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 