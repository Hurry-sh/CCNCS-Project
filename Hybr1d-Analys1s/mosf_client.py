
import logging
from typing import Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)
DEFAULT_SERVER = 'http://127.0.0.1:8000'

class MobSF:
    """Represents a MobSF instance."""

    def __init__(self, apikey: str, server: Optional[str] = None):
        """
        Initialize a MobSF instance.
        :param apikey: API key for authentication
        :param server: Server URL, defaults to http://127.0.0.1:8000
        """
        self._server = server.rstrip('/') if server else DEFAULT_SERVER
        self._apikey = apikey
        self.hash: Dict[str, Optional[str]] = {'hash': None}

    @property
    def server(self) -> str:
        return self._server

    @server.setter
    def server(self, value: str):
        self._server = value.rstrip('/')

    @property
    def apikey(self) -> str:
        return self._apikey

    @apikey.setter
    def apikey(self, value: str):
        self._apikey = value

    def upload(self, filename: str, file) -> Optional[Dict[str, Any]]:
        """
        Upload an app.
        :param filename: Name of the file
        :param file: File object to upload
        :return: Response from the server or None if upload failed
        """
        logger.debug(f"Uploading {filename} to {self.server}")
        multipart_data = {'file': (filename, file, 'application/octet-stream')}
        headers = {'Authorization': self.apikey}
        try:
            url = f'{self.server}/api/v1/upload'
            logger.debug(f"Request URL: {url}")
            logger.debug(f"Headers: {headers}")
            r = requests.post(url, files=multipart_data, headers=headers)
            r.raise_for_status()
            logger.debug(f"Response status code: {r.status_code}")
            logger.debug(f"Response content: {r.content}")
            response = r.json()
            self.hash = {"hash": response["hash"]}
            return response
        except requests.RequestException as e:
            logger.error(f"Upload failed: {str(e)}")
            logger.debug(f"Response content: {r.content}")
            return None

    def scan(self, data: Dict[str, str]) -> Dict[str, Any]:
        """
        Scan already uploaded file.
        If the file was not uploaded before you will have to do so first.
        :param data: Dictionary containing the hash of the file to scan
        :return: Scan results
        """
        logger.debug(f"Requesting {self.server} to scan {data['hash']}")
        headers = {'Authorization': self.apikey}
        r = requests.post(f'{self.server}/api/v1/scan', data=data, headers=headers)
        return r.json()

    def scans(self, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        """
        Show recent scans.
        :param page: Page number for pagination
        :param page_size: Number of items per page
        :return: List of recent scans
        """
        logger.debug(f'Requesting recent scans from {self.server}')
        payload = {'page': page, 'page_size': page_size}
        headers = {'Authorization': self.apikey}
        r = requests.get(f'{self.server}/api/v1/scans', params=payload, headers=headers)
        return r.json()

    def report_json(self, data: Dict[str, str]) -> Dict[str, Any]:
        """
        Retrieve JSON report of a scan.
        :param data: Dictionary containing the hash of the scan
        :return: JSON report of the scan
        """
        logger.debug(f'Requesting JSON report for scan {data["hash"]}')
        headers = {'Authorization': self.apikey}
        r = requests.post(f'{self.server}/api/v1/report_json', data=data, headers=headers)
        return r.json()

    def delete_scan(self, data: Dict[str, str]) -> Dict[str, Any]:
        """
        Delete a scan result.
        :param data: Dictionary containing the hash of the scan to delete
        :return: Response from the server
        """
        logger.debug(f'Requesting {self.server} to delete scan {data["hash"]}')
        headers = {'Authorization': self.apikey}
        r = requests.post(f'{self.server}/api/v1/delete_scan', data=data, headers=headers)
        return r.json()