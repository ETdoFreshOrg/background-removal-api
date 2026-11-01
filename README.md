# Background Removal API

Bria RMBG image processing with the existing web UI and HTTP upload endpoint.

## MCP

Streamable HTTP endpoint: `https://bgremove.etdofresh.com/mcp`.
Authenticate with an OAuth access token obtained through owner-key consent.
MCP fails closed if `MCP_OWNER_KEY` is not configured. `PUBLIC_ORIGIN` defaults to the hostname above.

- `background_removal_status`: model readiness and upload limit.
- `remove_background`: base64 PNG/JPEG/WebP input and filename; returns an
  inline transparent PNG at the original dimensions, plus job metadata.

The MCP reuses the existing processor and history. It cannot read paths on
a client's computer. Send file bytes as base64, up to the configured upload
limit (10 MB by default).

Transport tests do not download the model or import Torch:
`python -m unittest discover -s tests` with `mcp==1.26.0` and `httpx` installed.

## MCP owner-key OAuth

Background Removal MCP requires `MCP_OWNER_KEY`, with optional `PUBLIC_ORIGIN`
(default `https://bgremove.etdofresh.com`) and `OAUTH_DB` (default `/data/oauth.db`).
The compose stack persists OAuth state in `rmbg-oauth`. Use client ID
`bg-remove-claude` and scope `bg-remove.process`. `API_KEY` is not accepted at `/mcp`.

The owner key authorizes consent; clients receive 15-minute OAuth access tokens
and rotating refresh tokens. Grants last 30 days and survive restarts. Refresh
reuse or explicit revocation invalidates the consent. Only token hashes are stored.
Discover OAuth at `/.well-known/oauth-authorization-server` and
`/.well-known/oauth-protected-resource/mcp`. Consent and token endpoints are under
`/mcp/oauth`. Claude and ChatGPT metadata clients are supported.
