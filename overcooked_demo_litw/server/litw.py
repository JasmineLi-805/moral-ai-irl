import requests
import json
import logging
from datetime import datetime, timezone


class DataProxy:
    def __init__(self, litw_config):
        self.url_server = litw_config['url_server']
        self.url_auth = litw_config['url_auth']
        self.url_data = litw_config['url_data']
        self.study_id = litw_config['study_id']
        self.study_key = litw_config['study_key']
        self.litw_token = None

    @staticmethod
    def get_utc_timestamp():
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        return int(now.timestamp())

    def _is_token_valid(self):
        if isinstance(self.litw_token, dict):
            if 'access_token' in self.litw_token and 'expiration' in self.litw_token:
                now = DataProxy.get_utc_timestamp()
                if (self.litw_token['expiration'] - now) > 10:
                    return True
        return False

    def _refresh_litw_token(self):
        url = self.url_server + self.url_auth
        response = requests.get(url.format(self.study_id, self.study_key))
        if response.status_code == 200:
            try:
                self.litw_token = response.json()
            except Exception as e:
                logging.error('[LITW TOKEN]: {}'.format(e))
                self.litw_token = None
        else:
            self.litw_token = None

    def save_data(self, study_data, refresh_token=True):
        if not self._is_token_valid():
            if refresh_token:
                self._refresh_litw_token()
                return self.save_data(study_data, False)
            else:
                return False
        else:
            url = self.url_server + self.url_data.format(self.study_id)
            headers = {
                'Authorization': 'Bearer {}'.format(self.litw_token['access_token']),
                'content-type': 'application/json'
            }
            response = requests.put(url, data=json.dumps(study_data), headers=headers)
            if response.status_code == 200:
                return True
            else:
                logging.error('[LITW SAVE]: {}'.format(response))
                return False
