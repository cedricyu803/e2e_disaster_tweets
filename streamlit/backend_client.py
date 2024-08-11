# reference: https://fastapi.tiangolo.com/tutorial/security/

import logging
import urllib.parse

import requests


class BackendClient():
    def __init__(self,
                 endpoint_base: str = 'http://127.0.0.1:3100/',
                 ):
        '''Client for accessing API endpoint.

        Args:
            cloud_server_endpoint_base: str
        '''
        self._logger = logging.getLogger(__name__)
        self._endpoint_base = endpoint_base

        # if user endpoint is not present, start user chat engine
        # and update self._endpoint_base
        status = self.check_status()
        if status != 0:
            self._logger.error('Backend not running')

    def check_status(self,):
        endpoint = urllib.parse.urljoin(
            self._endpoint_base,
            'status/'
        )
        response = requests.get(
            url=endpoint,
        )
        return int(response.json()[0])

    def get_query(self, text: str):
        endpoint = urllib.parse.urljoin(
            self._endpoint_base,
            'query/'
        )
        response = requests.get(
            url=endpoint,
            params={'text': text},
        )
        return int(response.json()[0])
