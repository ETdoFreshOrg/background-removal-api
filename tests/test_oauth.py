import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from app.oauth import OwnerOAuth, PREFIX, CLIENT_ID, CLAUDE_REDIRECTS, SCOPE, COOKIE
from oauth_client import sign_in, begin


class OAuthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'oauth.db'
        self.provider = OwnerOAuth('fixture-owner-key', 'https://testserver', str(self.path))
        self.client = TestClient(self.provider.application(JSONResponse({'ok': True})), base_url='https://testserver')

    def tearDown(self):
        self.client.close()
        self.provider.db.close()
        self.temp.cleanup()

    def use(self, token):
        return self.client.post('/mcp', headers={'Authorization': 'Bearer ' + token})

    def refresh(self, token, resource='https://testserver/mcp'):
        return self.client.post(PREFIX + '/token', data={'grant_type': 'refresh_token', 'client_id': CLIENT_ID, 'refresh_token': token, 'resource': resource})

    def test_metadata_consent_csrf_and_legacy_key_rejection(self):
        response = self.use('fixture-owner-key')
        self.assertEqual(response.status_code, 401)
        self.assertIn('oauth-protected-resource/mcp', response.headers['www-authenticate'])
        self.assertEqual(self.client.get('/.well-known/oauth-authorization-server').json()['token_endpoint_auth_methods_supported'], ['none'])
        self.assertEqual(self.client.get('/.well-known/oauth-protected-resource/mcp').json()['resource'], 'https://testserver/mcp')
        started = begin(self.client)
        data = {'request': started['request'], 'owner_key': 'fixture-owner-key', 'decision': 'allow'}
        self.assertEqual(self.client.post(PREFIX + '/consent', data=data, headers={'Origin': 'https://evil.test'}).status_code, 403)
        cookie = self.client.cookies.get(COOKIE)
        self.client.cookies.clear()
        self.assertEqual(self.client.post(PREFIX + '/consent', data=data, headers={'Origin': 'https://testserver'}).status_code, 403)
        headers = {'Origin': 'https://testserver', 'Cookie': COOKIE + '=' + cookie}
        self.assertEqual(self.client.post(PREFIX + '/consent', data={**data, 'owner_key': 'wrong'}, headers=headers).status_code, 401)
        tokens, exchange = sign_in(self.client, '  fixture-owner-key\n')
        self.assertEqual(self.use(tokens['access_token']).status_code, 200)
        self.assertEqual(self.client.post(PREFIX + '/token', data=exchange).status_code, 400)

    def test_pkce_audience_scope_and_redirect_checks(self):
        for override in [{'client_id': 'unknown'}, {'redirect_uri': 'https://evil.test/callback'}, {'resource': 'https://other.test/mcp'}, {'scope': 'admin'}, {'code_challenge_method': 'plain'}]:
            self.assertNotEqual(begin(self.client, **override)['page'].status_code, 200)
        started = begin(self.client)
        consent = self.client.post(PREFIX + '/consent', data={'request': started['request'], 'owner_key': 'fixture-owner-key', 'decision': 'allow'}, headers={'Origin': 'https://testserver'}, follow_redirects=False)
        code = parse_qs(urlsplit(consent.headers['location']).query)['code'][0]
        exchange = {'grant_type': 'authorization_code', 'client_id': CLIENT_ID, 'redirect_uri': CLAUDE_REDIRECTS[0], 'resource': 'https://testserver/mcp', 'code': code, 'code_verifier': started['verifier']}
        for override in [{'code_verifier': 'wrong'}, {'resource': 'https://other.test/mcp'}, {'redirect_uri': 'https://evil.test/callback'}]:
            self.assertEqual(self.client.post(PREFIX + '/token', data={**exchange, **override}).status_code, 400)
        self.assertEqual(self.client.post(PREFIX + '/token', data=exchange).status_code, 200)

    def test_refresh_rotation_replay_and_revocation(self):
        tokens, _ = sign_in(self.client)
        self.assertEqual(self.refresh(tokens['refresh_token'], 'https://other.test/mcp').status_code, 400)
        fresh = self.refresh(tokens['refresh_token']).json()
        self.assertNotEqual(fresh['refresh_token'], tokens['refresh_token'])
        self.assertEqual(self.use(fresh['access_token']).status_code, 200)
        self.assertEqual(self.refresh(tokens['refresh_token']).status_code, 400)
        self.assertEqual(self.use(fresh['access_token']).status_code, 401)
        tokens, _ = sign_in(self.client)
        response = self.client.post(PREFIX + '/revoke', data={'client_id': CLIENT_ID, 'token': tokens['access_token']})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.use(tokens['access_token']).status_code, 401)
        self.assertEqual(self.refresh(tokens['refresh_token']).status_code, 400)

    def test_hashed_tokens_survive_restart_and_expire(self):
        tokens, _ = sign_in(self.client)
        self.provider.db.close()
        self.assertNotIn(tokens['access_token'].encode(), self.path.read_bytes())
        self.assertNotIn(tokens['refresh_token'].encode(), self.path.read_bytes())
        self.provider = OwnerOAuth('fixture-owner-key', 'https://testserver', str(self.path))
        self.client.close()
        self.client = TestClient(self.provider.application(JSONResponse({'ok': True})), base_url='https://testserver')
        self.assertEqual(self.use(tokens['access_token']).status_code, 200)
        now = self.provider.clock()
        self.provider.clock = lambda: now + 901
        self.assertEqual(self.use(tokens['access_token']).status_code, 401)
        self.assertEqual(self.refresh(tokens['refresh_token']).status_code, 200)
        self.provider.clock = lambda: now + 31 * 86400
        self.assertEqual(self.refresh(tokens['refresh_token']).status_code, 400)

    def test_missing_key_disables_mcp(self):
        provider = OwnerOAuth('', 'https://testserver')
        try:
            with TestClient(provider.application(JSONResponse({'ok': True})), base_url='https://testserver') as client:
                self.assertEqual(client.post('/mcp').status_code, 503)
        finally:
            provider.db.close()
