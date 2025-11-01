"""
PostgreSQL database connector for the Property Measurement RAG system.
"""
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from typing import Optional, Any, Dict, List
from contextlib import contextmanager
from .config import config

logger = logging.getLogger(__name__)

class PostgreSQLConnector:
    """
    PostgreSQL database connector with connection pooling and error handling.
    """

    def __init__(self, config_instance=None):
        """
        Initialize PostgreSQL connector.

        Args:
            config_instance: Optional config instance, uses global config if None
        """
        self.config_instance = config_instance or config
        self.pg_config = self.config_instance.get_postgresql_config()
        self.connection_pool = None
        self.db_available = False
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.pg_config['host'],
                port=self.pg_config['port'],
                dbname=self.pg_config['dbname'],
                user=self.pg_config['user'],
                password=self.pg_config['password']
            )
            self.db_available = True
            logger.info("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            self.db_available = False
            logger.warning(f"PostgreSQL connection failed - database operations will be skipped: {e}")
            logger.info("System will continue without database integration")

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a database connection.

        Yields:
            psycopg2 connection object
        """
        if not self.db_available:
            raise Exception("Database not available - operations will be skipped")

        conn = None
        try:
            conn = self.connection_pool.getconn()
            # Set search path to the configured schema
            with conn.cursor() as cursor:
                cursor.execute(f"SET search_path TO {self.pg_config['schema']}")
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """
        Context manager for getting a database cursor.

        Args:
            cursor_factory: Optional cursor factory (e.g., RealDictCursor)

        Yields:
            psycopg2 cursor object
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                # Set search path to the configured schema
                cursor.execute(f"SET search_path TO {self.pg_config['schema']}")
                yield cursor
            except Exception as e:
                conn.rollback()
                logger.error(f"Database cursor error: {e}")
                raise
            else:
                conn.commit()

    def execute_query(self, query: str, params: tuple = None, fetch: bool = True) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch and return results

        Returns:
            List of dictionaries containing query results, or None if fetch=False
        """
        try:
            with self.get_cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

    def execute_update(self, query: str, params: tuple = None) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Number of affected rows
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Update execution error: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def close_pool(self):
        """Close the connection pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")

    def __del__(self):
        """Cleanup on object destruction."""
        self.close_pool()


# Global connector instance
db_connector = PostgreSQLConnector()
