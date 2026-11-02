"""Single-owner OAuth for RMBG, using the MCP SDK's protocol handlers.

Consent, lifetimes and hashed durable token families mirror ETdoFreshOrg/usage-api
oauth/oauth.mjs. This native ASGI adapter needs no browser or proxy process.
"""
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit, parse_qsl, urlunsplit

import httpx
from mcp.server.auth.handlers.authorize import AuthorizationHandler
from mcp.server.auth.handlers.token import TokenHandler
from mcp.server.auth.middleware.client_auth import ClientAuthenticator, AuthenticationError
from mcp.server.auth.provider import AuthorizationCode, AccessToken, RefreshToken, AuthorizeError, TokenError
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route, Mount

SCOPE = 'bg-remove.process'
CLIENT_ID = 'bg-remove-claude'
CLAUDE_REDIRECTS = ['https://claude.ai/api/mcp/auth_callback', 'https://claude.com/api/mcp/auth_callback']
CLAUDE_CLIENTS = ['https://claude.ai/oauth/mcp-oauth-client-metadata', 'https://claude.com/oauth/mcp-oauth-client-metadata']
# ET Bot (etbot.etdofresh.com) names itself by its client metadata URL. It is pre-registered with its
# one exact callback, so the document is never fetched and no other ET Bot callback is accepted.
ETBOT_CLIENT_ID = 'https://etbot.etdofresh.com/oauth/client-metadata.json'
ETBOT_REDIRECTS = ['https://etbot.etdofresh.com/api/connectors/oauth/callback']
PREFIX = '/mcp/oauth'
COOKIE = 'bg_remove_oauth_csrf'
digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
random = lambda: secrets.token_urlsafe(32)


class OwnerOAuth:
    def __init__(self, owner_key, origin, db_path=':memory:', clock=time.time):
        if urlsplit(origin).scheme != 'https' or origin != origin.rstrip('/') or urlsplit(origin).path:
            raise ValueError('OAuth origin must be a bare HTTPS origin')
        self.owner_key, self.origin, self.resource, self.clock = owner_key, origin, origin + '/mcp', clock
        if db_path != ':memory:':
            Path(db_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        if db_path != ':memory:':
            os.chmod(db_path, 0o600)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS oauth (kind TEXT,key TEXT,expires REAL,value TEXT,PRIMARY KEY(kind,key))')
        self.lock = threading.RLock()
        self.pending, self.codes, self.clients, self.rates = {}, {}, {}, {}

    def prune(self):
        now = self.clock()
        with self.lock, self.db:
            self.db.execute('DELETE FROM oauth WHERE expires<=?', (now,))
        for mapping in [self.pending, self.codes, self.clients, self.rates]:
            for key, value in list(mapping.items()):
                if value['expires'] <= now:
                    del mapping[key]

    def get(self, kind, key):
        row = self.db.execute('SELECT value FROM oauth WHERE kind=? AND key=? AND expires>?', (kind, key, self.clock())).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, key, value):
        if self.db.execute('SELECT COUNT(*) FROM oauth WHERE kind=?', (kind,)).fetchone()[0] >= 1000:
            raise TokenError('invalid_grant', 'OAuth state capacity reached')
        self.db.execute('INSERT OR REPLACE INTO oauth VALUES(?,?,?,?)', (kind, key, value['expires'], json.dumps(value)))

    def delete_grant(self, grant):
        self.db.execute("DELETE FROM oauth WHERE kind='grant' AND key=?", (grant,))

    def client(self, client_id, redirects, name):
        return OAuthClientInformationFull(client_id=client_id, client_name=name, redirect_uris=redirects,
            token_endpoint_auth_method='none', grant_types=['authorization_code', 'refresh_token'], response_types=['code'], scope=SCOPE)

    async def get_client(self, client_id):
        if client_id == CLIENT_ID:
            return self.client(client_id, CLAUDE_REDIRECTS, 'Claude / Codex')
        if client_id == ETBOT_CLIENT_ID:
            return self.client(client_id, ETBOT_REDIRECTS, 'ET Bot')
        redirects, name = None, 'ChatGPT'
        if client_id in CLAUDE_CLIENTS:
            redirects, name = CLAUDE_REDIRECTS, 'Claude'
        elif client_id == 'https://chatgpt.com/oauth/client.json':
            redirects = ['https://chatgpt.com/connector_platform_oauth_redirect']
        else:
            match = re.fullmatch(r'https://chatgpt\.com/oauth/([A-Za-z0-9_-]{1,160})/client\.json', client_id)
            if match:
                redirects = ['https://chatgpt.com/connector/oauth/' + match[1]]
        if not redirects:
            return None
        self.prune()
        if client_id in self.clients:
            return self.clients[client_id]['client']
        try:
            async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
                async with client.stream('GET', client_id) as response:
                    if response.status_code != 200:
                        return None
                    body = b''
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) > 32768:
                            return None
            data = json.loads(body)
            allowed = [r for r in redirects if r in data.get('redirect_uris', [])]
            methods = data.get('token_endpoint_auth_methods_supported', [data.get('token_endpoint_auth_method')])
            if data.get('client_id') != client_id or not allowed or 'none' not in methods:
                return None
            result = self.client(client_id, allowed, name)
            if len(self.clients) < 1000:
                self.clients[client_id] = {'expires': self.clock() + 300, 'client': result}
            return result
        except (httpx.HTTPError, ValueError, TypeError):
            return None

    async def register_client(self, _client):
        raise NotImplementedError('Use a registered or supported metadata client')

    async def authorize(self, client, params):
        self.prune()
        if params.resource != self.resource:
            raise AuthorizeError('invalid_request', 'Resource must match this MCP endpoint')
        if params.scopes and params.scopes != [SCOPE]:
            raise AuthorizeError('invalid_scope', 'Unsupported scope')
        if not re.fullmatch(r'[A-Za-z0-9_-]{43}', params.code_challenge):
            raise AuthorizeError('invalid_request', 'S256 PKCE is required')
        if len(self.pending) >= 1000:
            raise AuthorizeError('temporarily_unavailable', 'Too many pending consents')
        request_id = random()
        self.pending[digest(request_id)] = {'client': client, 'params': params, 'expires': self.clock() + 300}
        return self.origin + PREFIX + '/consent?' + urlencode({'request': request_id})

    async def load_authorization_code(self, client, code):
        self.prune()
        record = self.codes.get(digest(code))
        if not record or record['client'].client_id != client.client_id:
            return None
        p = record['params']
        return AuthorizationCode(code=code, client_id=client.client_id, scopes=[SCOPE], expires_at=record['expires'],
            code_challenge=p.code_challenge, redirect_uri=p.redirect_uri, redirect_uri_provided_explicitly=True, resource=self.resource)

    def issue(self, client_id, grant_id):
        grant = self.get('grant', grant_id)
        if not grant:
            raise TokenError('invalid_grant', 'Consent expired or revoked')
        access, refresh = random(), random()
        ttl = min(900, int(grant['expires'] - self.clock()))
        self.put('access', digest(access), {'client_id': client_id, 'grant': grant_id, 'expires': self.clock() + ttl})
        self.put('refresh', digest(refresh), {'client_id': client_id, 'grant': grant_id, 'expires': grant['expires'], 'used': False})
        return OAuthToken(access_token=access, refresh_token=refresh, token_type='Bearer', expires_in=ttl, scope=SCOPE)

    async def exchange_authorization_code(self, client, code):
        record = self.codes.pop(digest(code.code), None)
        if not record or record['expires'] <= self.clock() or record['client'].client_id != client.client_id:
            raise TokenError('invalid_grant', 'Code expired or used')
        grant_id = random()
        with self.lock, self.db:
            self.put('grant', grant_id, {'expires': self.clock() + 30 * 86400})
            return self.issue(client.client_id, grant_id)

    async def load_refresh_token(self, client, token):
        record = self.get('refresh', digest(token))
        if not record or record['client_id'] != client.client_id:
            return None
        return RefreshToken(token=token, client_id=client.client_id, scopes=[SCOPE], expires_at=int(record['expires']))

    async def exchange_refresh_token(self, client, token, scopes):
        with self.lock, self.db:
            record = self.get('refresh', digest(token.token))
            if not record or record['client_id'] != client.client_id:
                raise TokenError('invalid_grant', 'Invalid refresh token')
            if record['used']:
                self.delete_grant(record['grant'])
                self.db.commit()  # Revocation must persist even though the exchange fails.
                raise TokenError('invalid_grant', 'Refresh token reuse revoked consent')
            if scopes != [SCOPE]:
                raise TokenError('invalid_scope', 'Cannot change consent')
            self.put('refresh', digest(token.token), {**record, 'used': True, 'expires': min(record['expires'], self.clock() + 86400)})
            return self.issue(client.client_id, record['grant'])

    async def load_access_token(self, token):
        record = self.get('access', digest(token))
        if not record or not self.get('grant', record['grant']):
            return None
        return AccessToken(token=token, client_id=record['client_id'], scopes=[SCOPE], expires_at=int(record['expires']), resource=self.resource)

    async def revoke_token(self, token):
        record = self.get('access', digest(token.token)) or self.get('refresh', digest(token.token))
        if record:
            with self.lock, self.db:
                self.delete_grant(record['grant'])

    def limited(self, request, name, limit):
        self.prune()
        key = (name, request.client.host if request.client else 'unknown')
        record = self.rates.setdefault(key, {'count': 0, 'expires': self.clock() + 900})
        record['count'] += 1
        return record['count'] > limit

    async def authorize_page(self, request):
        if self.limited(request, 'authorize', 100):
            return JSONResponse({'error': 'rate_limited'}, status_code=429)
        response = await AuthorizationHandler(self).handle(request)
        location = response.headers.get('location', '')
        if not location.startswith(self.origin + PREFIX + '/consent?'):
            if location:
                u = urlsplit(location)
                response.headers['location'] = urlunsplit((*u[:3], urlencode([*parse_qsl(u.query), ('iss', self.origin + '/')]), u.fragment))
            return response
        request_id = dict(parse_qsl(urlsplit(location).query))['request']
        csrf = random()
        self.pending[digest(request_id)]['csrf'] = digest(csrf)
        client = self.pending[digest(request_id)]['client']
        page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Authorize Background Removal</title><style>body{{font:18px system-ui;max-width:600px;margin:8vh auto;padding:24px}}input,button{{font:inherit;padding:10px;margin:10px 0;width:100%;box-sizing:border-box}}</style><h1>Authorize {html.escape(client.client_name or 'MCP client')}</h1><p>Allow background removal and model status checks. Uploaded images and results are saved in this server's history.</p><p>Permission: {SCOPE}. Access tokens last 15 minutes; consent lasts 30 days. The owner key is never sent to the connecting app.</p><form method="post" action="{PREFIX}/consent"><input type="hidden" name="request" value="{request_id}"><label>Background Removal MCP owner key<input type="password" name="owner_key" required autocomplete="off"></label><button name="decision" value="allow">Authorize access</button><button name="decision" value="deny" formnovalidate>Cancel</button></form></html>'''
        response = HTMLResponse(page, headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'same-origin', 'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"})
        response.set_cookie(COOKIE, csrf, secure=True, httponly=True, samesite='lax', path=PREFIX + '/consent', max_age=300)
        return response

    async def consent(self, request):
        if self.limited(request, 'consent', 30):
            return JSONResponse({'error': 'rate_limited'}, status_code=429)
        if len(await request.body()) > 4096:
            return JSONResponse({'error': 'Request too large'}, status_code=413)
        form = await request.form()
        record = self.pending.get(digest(str(form.get('request', ''))))
        if not record or request.headers.get('origin') != self.origin or not hmac.compare_digest(digest(request.cookies.get(COOKIE, '')), record.get('csrf', '')):
            return JSONResponse({'error': 'Invalid or expired consent'}, status_code=403)
        decision = form.get('decision')
        if decision != 'deny' and (decision != 'allow' or not hmac.compare_digest(str(form.get('owner_key', '')).strip().encode(), self.owner_key.encode())):
            return JSONResponse({'error': 'Authorization denied'}, status_code=401)
        del self.pending[digest(str(form['request']))]
        params = record['params']
        callback = urlsplit(str(params.redirect_uri))
        query = [*parse_qsl(callback.query), ('iss', self.origin + '/')]
        if params.state:
            query.append(('state', params.state))
        if decision == 'deny':
            query.append(('error', 'access_denied'))
        else:
            code = random()
            if len(self.codes) >= 1000:
                return JSONResponse({'error': 'capacity_reached'}, status_code=503)
            self.codes[digest(code)] = {**record, 'expires': self.clock() + 60}
            query.append(('code', code))
        response = RedirectResponse(urlunsplit((*callback[:3], urlencode(query), callback.fragment)), status_code=302)
        response.delete_cookie(COOKIE, path=PREFIX + '/consent', secure=True, httponly=True, samesite='lax')
        return response

    async def token(self, request):
        if self.limited(request, 'token', 300):
            return JSONResponse({'error': 'rate_limited'}, status_code=429)
        if len(await request.body()) > 8192:
            return JSONResponse({'error': 'Request too large'}, status_code=413)
        form = await request.form()
        # MCP Python SDK 1.26 parses resource but does not validate it in the
        # token handler. Enforce audience before either code or refresh exchange.
        if form.getlist('resource') != [self.resource]:
            return JSONResponse({'error': 'invalid_target'}, status_code=400)
        return await TokenHandler(self, ClientAuthenticator(self)).handle(request)

    async def revoke(self, request):
        if self.limited(request, 'revoke', 300):
            return JSONResponse({'error': 'rate_limited'}, status_code=429)
        if len(await request.body()) > 8192:
            return JSONResponse({'error': 'Request too large'}, status_code=413)
        # SDK 1.26's revocation model requires client_secret even for public
        # clients. Keep its client authentication, and accept RFC 7009's public
        # client form without inventing a client secret.
        try:
            client = await ClientAuthenticator(self).authenticate_request(request)
        except AuthenticationError:
            return JSONResponse({'error': 'unauthorized_client'}, status_code=401)
        form = await request.form()
        raw = form.get('token')
        if not isinstance(raw, str) or not raw or form.get('token_type_hint') not in (None, 'access_token', 'refresh_token'):
            return JSONResponse({'error': 'invalid_request'}, status_code=400)
        token = await self.load_access_token(raw) or await self.load_refresh_token(client, raw)
        if token and token.client_id == client.client_id:
            await self.revoke_token(token)
        return Response(status_code=200, headers={'Cache-Control': 'no-store', 'Pragma': 'no-cache'})

    async def metadata(self, _request):
        return JSONResponse({'issuer': self.origin + '/', 'authorization_endpoint': self.origin + PREFIX + '/authorize',
            'token_endpoint': self.origin + PREFIX + '/token', 'revocation_endpoint': self.origin + PREFIX + '/revoke',
            'scopes_supported': [SCOPE], 'response_types_supported': ['code'], 'grant_types_supported': ['authorization_code', 'refresh_token'],
            'code_challenge_methods_supported': ['S256'], 'token_endpoint_auth_methods_supported': ['none'], 'revocation_endpoint_auth_methods_supported': ['none'],
            'client_id_metadata_document_supported': True, 'authorization_response_iss_parameter_supported': True})

    async def resource_metadata(self, _request):
        return JSONResponse({'resource': self.resource, 'authorization_servers': [self.origin + '/'], 'scopes_supported': [SCOPE], 'resource_name': 'Background Removal'})

    def application(self, mcp_app):
        if not self.owner_key:
            async def disabled(_request):
                return JSONResponse({'error': 'MCP owner-key OAuth is not configured'}, status_code=503)
            return Starlette(routes=[Route('/{path:path}', disabled, methods=['GET', 'POST', 'DELETE'])])
        routes = [Route('/.well-known/oauth-authorization-server', self.metadata),
            Route('/.well-known/oauth-protected-resource', self.resource_metadata),
            Route('/.well-known/oauth-protected-resource/mcp', self.resource_metadata)]
        for suffix, handler, methods in [('/authorize', self.authorize_page, ['GET']), ('/consent', self.consent, ['POST']), ('/token', self.token, ['POST']), ('/revoke', self.revoke, ['POST'])]:
            routes.append(Route(PREFIX + suffix, handler, methods=methods))
        routes.append(Mount('/', app=RequireOAuth(mcp_app, self)))
        return Starlette(routes=routes)


class RequireOAuth:
    def __init__(self, app, provider):
        self.app, self.provider = app, provider

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            headers = dict(scope.get('headers', []))
            if headers.get(b'origin') and headers[b'origin'].decode('latin1') != self.provider.origin:
                return await JSONResponse({'error': 'Origin not allowed'}, status_code=403)(scope, receive, send)
            match = re.fullmatch(r'Bearer (\S+)', headers.get(b'authorization', b'').decode('latin1'))
            auth = await self.provider.load_access_token(match[1]) if match else None
            if not auth or auth.resource != self.provider.resource or SCOPE not in auth.scopes:
                challenge = f'Bearer resource_metadata="{self.provider.origin}/.well-known/oauth-protected-resource/mcp", scope="{SCOPE}"'
                return await JSONResponse({'error': 'OAuth authorization required'}, status_code=401, headers={'WWW-Authenticate': challenge, 'Cache-Control': 'no-store'})(scope, receive, send)
        return await self.app(scope, receive, send)
