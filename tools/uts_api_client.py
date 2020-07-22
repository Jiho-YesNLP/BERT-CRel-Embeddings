""" UTS (UMLS Terminology Services) API client
REST documentation: https://documentation.uts.nlm.nih.gov/rest/home.html
"""
import code
import os
import json
from pathlib import Path
import csv
import time

import requests
from lxml.html import fromstring


class UtsClient:
    """All the UTS REST API requests are handled through this client"""
    apikey_file = Path(__file__).resolve().parent / 'uts.key'

    def __init__(self):
        if not os.path.exists(self.apikey_file):
            raise RuntimeError("API key file not exists [{}]"
                               .format(self.apikey_file))
        self.apikey = open(self.apikey_file).read().rstrip()
        self.service = "http://umlsks.nlm.nih.gov"
        self.headers = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
            "User-Agent": "python"
        }
        self.tgt = None
        self.base_uri = "https://uts-ws.nlm.nih.gov"
        self.version = "current"

    def gettgt(self):
        """Retrieve a ticket granting ticket"""
        auth_uri = "https://utslogin.nlm.nih.gov"
        params = {"apikey": self.apikey}
        auth_endpoint = "/cas/v1/api-key"
        r = requests.post(auth_uri + auth_endpoint, data=params,
                          headers=self.headers)
        response = fromstring(r.text)
        # extract the entire URL needed from the HTML form (action attribute)
        # returned - looks similar to
        # https://utslogin.nlm.nih.gov/cas/v1/tickets/TGT-36471-aYqNLN2rFIJPXKzxwdTNC5ZT7z3B3cTAKfSc5ndHQcUxeaDOLN-cas
        # we make a POST call to this URL in the getst method
        self.tgt = response.xpath('//form/@action')[0]

    def getst(self):
        """Request a single-use service ticket"""
        if self.tgt is None:
            self.gettgt()
        params = {"service": self.service}
        r = requests.post(self.tgt, data=params, headers=self.headers)
        return r.text

    def query_get(self, endpoint, query):
        r = requests.get(self.base_uri+endpoint, params=query)
        if r.status_code == 404:
            return
        return json.loads(r.text)

    def get_mesh(self, cui=None, phrase=None):
        # First, try searching by CUI
        if cui is not None:
            endpoint = '/rest/content/{}/CUI/{}/atoms'.format(self.version, cui)
            params = {'ticket': self.getst(), 'sabs': 'MSH', 'pageSize': 5}
            rst = self.query_get(endpoint, params)
            if 'result' in rst:
                for r in rst['result']:
                    if 'code' in r:
                        toks = r['code'].split('/')
                        if toks[-2] == 'MSH':
                            return toks[-1]
        if phrase is not None:
            endpoint = "/rest/search/current"
            params = {
                'ticket': self.getst(),
                'sabs': 'MSH',
                'pageSize': 5,
                'string': phrase,
                'returnIdType': 'sourceUi',
                'language': 'ENG'
            }
            rst = self.query_get(endpoint, params)
            if 'result' in rst:
                for r in rst['result']['results']:
                    return r['ui']
        return


if __name__ == '__main__':
    fin = 'data/eval/EHR-Rel.tsv'
    fout = 'data/eval/EHR-Rel-mesh.csv'
    uts = UtsClient()
    cache = dict()

    with open(fin) as f, open(fout, 'w') as fo:
        csv_reader = csv.reader(f, delimiter='\t')
        next(csv_reader)
        for i, rec in enumerate(csv_reader):
            c1_label, c2_label, c1_cui, c2_cui = rec[1], rec[3], rec[10], rec[11]
            if c1_cui in cache:
                m1 = cache[c1_cui]
            else:
                m1 = uts.get_mesh(c1_cui, c1_label)
                cache[c1_cui] = m1
                time.sleep(0.5)
            if c2_cui in cache:
                m2 = cache[c2_cui]
            else:
                m2 = uts.get_mesh(c2_cui, c2_label)
                cache[c2_cui] = m2
                time.sleep(0.5)
            fo.write('{}, {}, {}, {}, {}\n'
                     ''.format(c1_label, c2_label, m1, m2, rec[9]))
            print('{}/ {}, {}, {}, {}, {}'
                     ''.format(i, c1_label, c2_label, m1, m2, rec[9]))