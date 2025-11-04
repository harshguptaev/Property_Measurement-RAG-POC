#!/usr/bin/env python3
"""
Address to Lat/Long Converter
Converts addresses to latitude and longitude coordinates using EagleView geocoding API
"""

import requests
import json
import logging
from typing import Dict, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AddressToLatLong:
    def __init__(self):
        """Initialize the address to lat/long converter"""
        self.auth_url = "https://api.cmh.platform-prod.evinternal.net/auth-service/v1/token"
        self.geocoder_url = "https://api.cmh.platform-prod.evinternal.net/pdw/geocoder/v2/forward"
        
        # Basic auth credentials
        self.username = "0oaol0mj7qEnh4JHN2p7"
        self.password = "Uwzl4O4zr2RWKHU_xmIArMrRn4pB7CiFw34Izw_55a9Yg2rg0ic4z4bUIRq_gM7F"
        
        # Token cache
        self.access_token = None
        self.token_expires_at = None
    
    def get_access_token(self) -> Optional[str]:
        """
        Get access token for API authentication
        
        Returns:
            Access token string or None if failed
        """
        try:
            # Check if we have a valid cached token
            if self.access_token and self.token_expires_at:
                import time
                if time.time() < self.token_expires_at - 300:  # 5 minutes buffer
                    logger.info("Using cached access token")
                    return self.access_token
            
            logger.info("Requesting new access token...")
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
                'Authorization': f'Basic {self._encode_basic_auth()}',
                'Cookie': 'AWSALBTG=cAB2kn7BsAlo9iLN75KFFul+FwGpYGXjBYpRitEhmBdVjkbqkGC7noi61IeQSGU336igp1IWJ6HPV9R+fE+8ylnKdrPzjlk9x0wkspZLR2E4hG42uSNofYZ5oTsiBUsMDr/yzwCBDdmFVFfAhyh/jfNRK/i2jmbC+SLIwfwrCMqF; AWSALBTGCORS=cAB2kn7BsAlo9iLN75KFFul+FwGpYGXjBYpRitEhmBdVjkbqkGC7noi61IeQSGU336igp1IWJ6HPV9R+fE+8ylnKdrPzjlk9x0wkspZLR2E4hG42uSNofYZ5oTsiBUsMDr/yzwCBDdmFVFfAhyh/jfNRK/i2jmbC+SLIwfwrCMqF'
            }
            
            data = 'grant_type=client_credentials'
            
            response = requests.post(self.auth_url, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            
            # Calculate expiration time (expires_in is in seconds)
            import time
            self.token_expires_at = time.time() + token_data['expires_in']
            
            logger.info("✅ Access token obtained successfully")
            return self.access_token
            
        except Exception as e:
            logger.error(f"❌ Failed to get access token: {str(e)}")
            return None
    
    def _encode_basic_auth(self) -> str:
        """Encode username and password for basic auth"""
        import base64
        credentials = f"{self.username}:{self.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return encoded_credentials
    
    def geocode_address(self, address: str) -> Optional[Dict]:
        """
        Convert address to latitude and longitude
        
        Args:
            address: The address string to geocode
            
        Returns:
            Dictionary with lat/long and address info or None if failed
        """
        try:
            # Get access token
            token = self.get_access_token()
            if not token:
                logger.error("❌ Cannot get access token for geocoding")
                return None
            
            logger.info(f"🔄 Geocoding address: {address}")
            
            # Prepare request
            headers = {
                'Authorization': f'Bearer {token}',
                'Cookie': 'AWSALBTG=zrOdh8N3FtdvdcWObl/lMFsxgEJYvnFd/yU3yvExUd84A2zOWdBwFfCAXrIS0ldEgvJgklbQOhy50UU/B5f8gb1zXPy8yBI7Xm+N/K5oTuL1lwkkyulw9JKURnIcRJ2Jdsc0dcJuBCEwvZCIqIrdseuO+QH5CoGu4idCN4B1SHzn; AWSALBTGCORS=zrOdh8N3FtdvdcWObl/lMFsxgEJYvnFd/yU3yvExUd84A2zOWdBwFfCAXrIS0ldEgvJgklbQOhy50UU/B5f8gb1zXPy8yBI7Xm+N/K5oTuL1lwkkyulw9JKURnIcRJ2Jdsc0dcJuBCEwvZCIqIrdseuO+QH5CoGu4idCN4B1SHzn'
            }
            
            params = {'address': address}
            
            response = requests.get(self.geocoder_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Check if geocoding was successful
            if result.get('status', {}).get('code') == 1101:
                lat = result['lat']
                lon = result['lon']
                
                # Round to 6 decimal places as requested
                lat_rounded = round(lat, 6)
                lon_rounded = round(lon, 6)
                
                logger.info(f"✅ Geocoding successful: {lat_rounded}, {lon_rounded}")
                
                return {
                    'address': result['address'],
                    'lat': lat_rounded,
                    'lon': lon_rounded,
                    'original_lat': lat,
                    'original_lon': lon,
                    'status': result['status'],
                    'geocoder': result.get('geocoder', ''),
                    'input': result.get('input', {})
                }
            else:
                logger.error(f"❌ Geocoding failed: {result.get('status', {})}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error geocoding address '{address}': {str(e)}")
            return None
    
    def get_lat_lon(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Get latitude and longitude as a tuple
        
        Args:
            address: The address string to geocode
            
        Returns:
            Tuple of (latitude, longitude) or None if failed
        """
        result = self.geocode_address(address)
        if result:
            return (result['lat'], result['lon'])
        return None

if __name__ == "__main__":
    # Test the geocoding functionality
    converter = AddressToLatLong()
    
    test_address = "8440 Berry Brush Ln, Houston, TX 77022"
    print(f"Testing geocoding for: {test_address}")
    
    result = converter.geocode_address(test_address)
    if result:
        print(f"✅ Success: {result['lat']}, {result['lon']}")
        print(f"Address: {result['address']}")
    else:
        print("❌ Geocoding failed")
