import base64
import hashlib
import re
import secrets
from urllib.parse import urlsplit, parse_qs

from app.oauth import CLIENT_ID, PREFIX, SCOPE, CLAUDE_REDIRECTS


def begin(client, origin='https://testserver', **overrides):
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
    params = dict(client_id=CLIENT_ID, redirect_uri=CLAUDE_REDIRECTS[0], resource=origin + '/mcp', response_type='code',
                  code_challenge=challenge, code_challenge_method='S256', state='test-state', scope=SCOPE)
    params.update(overrides)
    page = client.get(PREFIX + '/authorize', params=params, follow_redirects=False)
    match = re.search(r'name="request" value="([^"]+)"', page.text)
    return {'page': page, 'verifier': verifier, 'request': match[1] if match else None}


def sign_in(client, owner='fixture-owner-key', origin='https://testserver'):
    started = begin(client, origin)
    assert started['page'].status_code == 200, started['page'].text
    response = client.post(PREFIX + '/consent', data={'request': started['request'], 'owner_key': owner, 'decision': 'allow'}, headers={'Origin': origin}, follow_redirects=False)
    assert response.status_code == 302, response.text
    callback = parse_qs(urlsplit(response.headers['location']).query)
    assert callback['iss'] == [origin + '/'] and callback['state'] == ['test-state']
    exchange = dict(grant_type='authorization_code', code=callback['code'][0], code_verifier=started['verifier'], client_id=CLIENT_ID,
                    redirect_uri=CLAUDE_REDIRECTS[0], resource=origin + '/mcp')
    response = client.post(PREFIX + '/token', data=exchange)
    assert response.status_code == 200, response.text
    return response.json(), exchange
