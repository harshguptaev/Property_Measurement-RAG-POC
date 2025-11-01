"""
Data Access Object (DAO) for property_rag_status table operations.
Provides comprehensive CRUD operations and utility functions for the property_rag_status table.
"""
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from .db_connector import db_connector

logger = logging.getLogger(__name__)

class PropertyRAGStatusDAO:
    """Data Access Object for property_rag_status table operations."""

    @staticmethod
    def insert_record(address: str, lat: Optional[float] = None, lng: Optional[float] = None,
                     report_id: Optional[str] = None, product_id: Optional[str] = None,
                     no_of_facets: Optional[int] = None, predominant_pitch: Optional[str] = None) -> int:
        """
        Insert a new record into property_rag_status table.

        Args:
            address: Property address (will be trimmed)
            lat: Latitude coordinate
            lng: Longitude coordinate
            report_id: Report ID from chunks
            product_id: Product ID (hardcoded value)
            no_of_facets: Number of facets
            predominant_pitch: Predominant pitch information

        Returns:
            The ID of the inserted record
        """
        if not db_connector.db_available:
            logger.warning("Database not available - skipping insert_record operation")
            return -1  # Return dummy ID when DB is not available

        try:
            insert_query = """
            INSERT INTO truedesigndemo.property_rag_status
            (address, lat, lng, reportId, productId, no_of_facets, predominant_pitch)
            VALUES (TRIM(%s), %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """

            params = (address, lat, lng, report_id, product_id, no_of_facets, predominant_pitch)
            result = db_connector.execute_query(insert_query, params)

            if result and len(result) > 0:
                record_id = result[0]['id']
                logger.info(f"Inserted new property_rag_status record with ID: {record_id}")
                return record_id
            else:
                raise Exception("No ID returned from insert operation")

        except Exception as e:
            logger.error(f"Error inserting record: {e}")
            raise

    @staticmethod
    def get_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a record by its ID.

        Args:
            record_id: The record ID

        Returns:
            Record dictionary or None if not found
        """
        try:
            query = "SELECT * FROM truedesigndemo.property_rag_status WHERE id = %s;"
            result = db_connector.execute_query(query, (record_id,))
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting record by ID {record_id}: {e}")
            raise

    @staticmethod
    def get_records_by_report_id(report_id: str) -> List[Dict[str, Any]]:
        """
        Get all records for a specific report ID.

        Args:
            report_id: The report ID

        Returns:
            List of record dictionaries
        """
        if not db_connector.db_available:
            logger.warning("Database not available - returning empty list for get_records_by_report_id")
            return []

        try:
            query = "SELECT * FROM truedesigndemo.property_rag_status WHERE reportId = %s ORDER BY id;"
            result = db_connector.execute_query(query, (report_id,))
            return result or []
        except Exception as e:
            logger.error(f"Error getting records by report ID {report_id}: {e}")
            raise

    @staticmethod
    def update_chunking_status(record_id: int, status: str) -> bool:
        """
        Update the chunking status for a record.

        Args:
            record_id: The record ID
            status: New chunking status (pending/processing/completed)

        Returns:
            True if update successful
        """
        if not db_connector.db_available:
            logger.warning("Database not available - skipping update_chunking_status operation")
            return False

        try:
            query = "UPDATE truedesigndemo.property_rag_status SET chunking_status = %s WHERE id = %s;"
            affected_rows = db_connector.execute_update(query, (status, record_id))
            return affected_rows > 0
        except Exception as e:
            logger.error(f"Error updating chunking status for record {record_id}: {e}")
            raise

    @staticmethod
    def update_vector_save_status(record_id: int, status: str) -> bool:
        """
        Update the vector save status for a record.

        Args:
            record_id: The record ID
            status: New vector save status (pending/processing/completed)

        Returns:
            True if update successful
        """
        if not db_connector.db_available:
            logger.warning("Database not available - skipping update_vector_save_status operation")
            return False

        try:
            query = "UPDATE truedesigndemo.property_rag_status SET vector_save_status = %s WHERE id = %s;"
            affected_rows = db_connector.execute_update(query, (status, record_id))
            return affected_rows > 0
        except Exception as e:
            logger.error(f"Error updating vector save status for record {record_id}: {e}")
            raise

    @staticmethod
    def update_final_status(record_id: int, status: str, reason: Optional[str] = None) -> bool:
        """
        Update the final status for a record.

        Args:
            record_id: The record ID
            status: Final status (success/failure/pending)
            reason: Failure reason (required if status is 'failure')

        Returns:
            True if update successful
        """
        if not db_connector.db_available:
            logger.warning("Database not available - skipping update_final_status operation")
            return False

        try:
            if status == 'failure' and not reason:
                raise ValueError("Reason is required when status is 'failure'")

            query = "UPDATE truedesigndemo.property_rag_status SET final_status = %s, reason = %s WHERE id = %s;"
            affected_rows = db_connector.execute_update(query, (status, reason, record_id))
            return affected_rows > 0
        except Exception as e:
            logger.error(f"Error updating final status for record {record_id}: {e}")
            raise

    @staticmethod
    def update_processing_details(record_id: int, lat: Optional[float] = None,
                                 lng: Optional[float] = None, no_of_facets: Optional[int] = None,
                                 predominant_pitch: Optional[str] = None) -> bool:
        """
        Update processing details for a record.

        Args:
            record_id: The record ID
            lat: Latitude coordinate
            lng: Longitude coordinate
            no_of_facets: Number of facets
            predominant_pitch: Predominant pitch information

        Returns:
            True if update successful
        """
        if not db_connector.db_available:
            logger.warning("Database not available - skipping update_processing_details operation")
            return False

        try:
            query = """
            UPDATE truedesigndemo.property_rag_status
            SET lat = COALESCE(%s, lat),
                lng = COALESCE(%s, lng),
                no_of_facets = COALESCE(%s, no_of_facets),
                predominant_pitch = COALESCE(%s, predominant_pitch)
            WHERE id = %s;
            """
            params = (lat, lng, no_of_facets, predominant_pitch, record_id)
            affected_rows = db_connector.execute_update(query, params)
            return affected_rows > 0
        except Exception as e:
            logger.error(f"Error updating processing details for record {record_id}: {e}")
            raise

    @staticmethod
    def get_status_counts() -> Dict[str, int]:
        """
        Get counts of records by final status.

        Returns:
            Dictionary with status counts
        """
        try:
            query = """
            SELECT final_status, COUNT(*) as count
            FROM truedesigndemo.property_rag_status
            GROUP BY final_status;
            """
            result = db_connector.execute_query(query)
            return {row['final_status']: row['count'] for row in result} if result else {}
        except Exception as e:
            logger.error(f"Error getting status counts: {e}")
            raise

    @staticmethod
    def get_failed_records() -> List[Dict[str, Any]]:
        """
        Get all records with failed final status.

        Returns:
            List of failed records with their failure reasons
        """
        try:
            query = """
            SELECT * FROM truedesigndemo.property_rag_status
            WHERE final_status = 'failure'
            ORDER BY id;
            """
            result = db_connector.execute_query(query)
            return result or []
        except Exception as e:
            logger.error(f"Error getting failed records: {e}")
            raise

    @staticmethod
    def get_pending_records() -> List[Dict[str, Any]]:
        """
        Get all records with pending final status.

        Returns:
            List of pending records
        """
        try:
            query = """
            SELECT * FROM truedesigndemo.property_rag_status
            WHERE final_status = 'pending'
            ORDER BY id;
            """
            result = db_connector.execute_query(query)
            return result or []
        except Exception as e:
            logger.error(f"Error getting pending records: {e}")
            raise

    @staticmethod
    def delete_record(record_id: int) -> bool:
        """
        Delete a record by its ID.

        Args:
            record_id: The record ID to delete

        Returns:
            True if deletion successful
        """
        try:
            query = "DELETE FROM truedesigndemo.property_rag_status WHERE id = %s;"
            affected_rows = db_connector.execute_update(query, (record_id,))
            return affected_rows > 0
        except Exception as e:
            logger.error(f"Error deleting record {record_id}: {e}")
            raise

    @staticmethod
    def get_all_records(limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all records with optional pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of all records
        """
        try:
            query = "SELECT * FROM truedesigndemo.property_rag_status ORDER BY id"
            params = []

            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)

            if offset is not None:
                query += " OFFSET %s"
                params.append(offset)

            query += ";"

            result = db_connector.execute_query(query, tuple(params) if params else None)
            return result or []
        except Exception as e:
            logger.error(f"Error getting all records: {e}")
            raise

    @staticmethod
    def get_records_by_address_pattern(address_pattern: str) -> List[Dict[str, Any]]:
        """
        Search records by address pattern (case-insensitive).

        Args:
            address_pattern: Pattern to search for in addresses

        Returns:
            List of matching records
        """
        try:
            query = """
            SELECT * FROM truedesigndemo.property_rag_status
            WHERE LOWER(address) LIKE LOWER(%s)
            ORDER BY id;
            """
            result = db_connector.execute_query(query, (f'%{address_pattern}%',))
            return result or []
        except Exception as e:
            logger.error(f"Error searching records by address pattern '{address_pattern}': {e}")
            raise

    @staticmethod
    def get_processing_stats() -> Dict[str, Any]:
        """
        Get comprehensive processing statistics.

        Returns:
            Dictionary with various statistics
        """
        try:
            stats = {
                'total_records': 0,
                'final_status_breakdown': {},
                'chunking_status_breakdown': {},
                'vector_save_status_breakdown': {},
                'records_with_coordinates': 0,
                'records_with_facets': 0,
                'average_facets': 0
            }

            # Total records
            total_query = "SELECT COUNT(*) as count FROM truedesigndemo.property_rag_status;"
            total_result = db_connector.execute_query(total_query)
            stats['total_records'] = total_result[0]['count'] if total_result else 0

            # Status breakdowns
            stats['final_status_breakdown'] = PropertyRAGStatusDAO.get_status_counts()

            # Chunking status breakdown
            chunking_query = """
            SELECT chunking_status, COUNT(*) as count
            FROM truedesigndemo.property_rag_status
            GROUP BY chunking_status;
            """
            chunking_result = db_connector.execute_query(chunking_query)
            stats['chunking_status_breakdown'] = {row['chunking_status']: row['count'] for row in chunking_result} if chunking_result else {}

            # Vector save status breakdown
            vector_query = """
            SELECT vector_save_status, COUNT(*) as count
            FROM truedesigndemo.property_rag_status
            GROUP BY vector_save_status;
            """
            vector_result = db_connector.execute_query(vector_query)
            stats['vector_save_status_breakdown'] = {row['vector_save_status']: row['count'] for row in vector_result} if vector_result else {}

            # Records with coordinates
            coord_query = "SELECT COUNT(*) as count FROM truedesigndemo.property_rag_status WHERE lat IS NOT NULL AND lng IS NOT NULL;"
            coord_result = db_connector.execute_query(coord_query)
            stats['records_with_coordinates'] = coord_result[0]['count'] if coord_result else 0

            # Records with facets
            facets_query = "SELECT COUNT(*) as count FROM truedesigndemo.property_rag_status WHERE no_of_facets IS NOT NULL;"
            facets_result = db_connector.execute_query(facets_query)
            stats['records_with_facets'] = facets_result[0]['count'] if facets_result else 0

            # Average facets
            avg_query = "SELECT AVG(no_of_facets) as avg FROM truedesigndemo.property_rag_status WHERE no_of_facets IS NOT NULL;"
            avg_result = db_connector.execute_query(avg_query)
            stats['average_facets'] = float(avg_result[0]['avg']) if avg_result and avg_result[0]['avg'] else 0

            return stats

        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            raise


# Convenience functions for easy access
def insert_property_record(address: str, **kwargs) -> int:
    """Convenience function to insert a new property record."""
    return PropertyRAGStatusDAO.insert_record(address, **kwargs)

def get_property_record(record_id: int) -> Optional[Dict[str, Any]]:
    """Convenience function to get a property record by ID."""
    return PropertyRAGStatusDAO.get_record_by_id(record_id)

def update_chunking_status(record_id: int, status: str) -> bool:
    """Convenience function to update chunking status."""
    return PropertyRAGStatusDAO.update_chunking_status(record_id, status)

def update_vector_status(record_id: int, status: str) -> bool:
    """Convenience function to update vector save status."""
    return PropertyRAGStatusDAO.update_vector_save_status(record_id, status)

def mark_success(record_id: int) -> bool:
    """Convenience function to mark a record as successful."""
    return PropertyRAGStatusDAO.update_final_status(record_id, 'success')

def mark_failure(record_id: int, reason: str) -> bool:
    """Convenience function to mark a record as failed with reason."""
    return PropertyRAGStatusDAO.update_final_status(record_id, 'failure', reason)

def get_processing_statistics() -> Dict[str, Any]:
    """Convenience function to get comprehensive processing statistics."""
    return PropertyRAGStatusDAO.get_processing_stats()
