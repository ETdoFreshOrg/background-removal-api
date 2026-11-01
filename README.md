# Background Removal API

Bria RMBG image processing with the existing web UI and HTTP upload endpoint.

## MCP

Streamable HTTP endpoint: `https://bgremove.etdofresh.com/mcp`.
Authenticate with `Authorization: Bearer <API_KEY>`. MCP fails closed if no
API key is configured. `PUBLIC_ORIGIN` defaults to the hostname above.

- `background_removal_status`: model readiness and upload limit.
- `remove_background`: base64 PNG/JPEG/WebP input and filename; returns an
  inline transparent PNG at the original dimensions, plus job metadata.

The MCP reuses the existing processor and history. It cannot read paths on
a client's computer. Send file bytes as base64, up to the configured upload
limit (10 MB by default).

Transport tests do not download the model or import Torch:
`python -m unittest discover -s tests` with `mcp==1.26.0` and `httpx` installed.
