import base64
import unittest
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.testclient import TestClient
from app.mcp_server import create_mcp, RequireMcpKey


class McpTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        async def process(data, filename):
            self.calls.append((data, filename))
            return b'PNG-result', '{"job_id":"test"}'

        self.server = create_mcp(process, lambda: {'status': 'ok'}, 1)

        @asynccontextmanager
        async def lifespan(app):
            async with self.server.session_manager.run():
                yield

        app = Starlette(routes=[Mount('/', app=RequireMcpKey(self.server.streamable_http_app(), 'test-key'))], lifespan=lifespan)
        self.client = TestClient(app).__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def rpc(self, method, params=None, key='test-key'):
        return self.client.post('/mcp', headers={'Authorization': 'Bearer '+key, 'Accept': 'application/json, text/event-stream'}, json={'jsonrpc':'2.0', 'id':1, 'method':method, 'params':params or {}})

    def test_auth_and_discovery(self):
        self.assertEqual(self.rpc('tools/list', key='wrong').status_code, 401)
        result = self.rpc('tools/list').json()['result']
        self.assertEqual({t['name'] for t in result['tools']}, {'remove_background','background_removal_status'})

    def test_processor_and_image_result(self):
        r = self.rpc('tools/call', {'name':'remove_background', 'arguments':{'image_base64':base64.b64encode(b'input').decode(), 'filename':'input.png'}}).json()['result']
        self.assertFalse(r.get('isError'))
        self.assertEqual(self.calls, [(b'input', 'input.png')])
        self.assertTrue(any(c.get('type') == 'image' for c in r['content']))

    def test_invalid_input_never_reaches_processor(self):
        for args in [{'image_base64':'!!!!'}, {'image_base64':'aGVsbG8=', 'filename':'../secret'}]:
            r = self.rpc('tools/call', {'name':'remove_background','arguments':args}).json()['result']
            self.assertTrue(r['isError'])
        self.assertEqual(self.calls, [])
