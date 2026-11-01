"""Authenticated Streamable HTTP tools over the existing image processor."""
import base64
import binascii
import os
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import Field


def create_mcp(process_image, get_health, max_upload_mb=10):
    origin = os.getenv("PUBLIC_ORIGIN", "https://bgremove.etdofresh.com")
    server = FastMCP(
        "background-removal", stateless_http=True, json_response=True,
        streamable_http_path="/mcp",
        instructions="Remove image backgrounds with Bria RMBG. Supply image bytes as base64; paths on the caller's computer are not accessible to this remote server. The result is a PNG with transparency at the original dimensions.",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[origin.split('://', 1)[-1], "127.0.0.1:*", "localhost:*", "testserver"],
            allowed_origins=[origin],
        ),
    )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def background_removal_status() -> dict:
        """Check model readiness, device and upload limit without processing an image."""
        return {**get_health(), "max_upload_mb": max_upload_mb}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
    async def remove_background(
        image_base64: Annotated[str, Field(min_length=4, max_length=14 * 1024 * 1024)],
        filename: Annotated[str, Field(min_length=1, max_length=255)] = "image.png",
    ) -> list:
        """Remove a PNG/JPEG/WebP image's background. Return the transparent PNG inline. Input is base64 file content, not a filename or data URI."""
        if any(c in filename for c in ('/', '\\', '\0')):
            raise ValueError("filename must be a name without a path")
        try:
            data = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("image_base64 is not valid base64") from None
        if len(data) > max_upload_mb * 1024 * 1024:
            raise ValueError("Image exceeds upload limit")
        png, metadata = await process_image(data, filename)
        return [TextContent(type="text", text=metadata), ImageContent(type="image", data=base64.b64encode(png).decode(), mimeType="image/png")]

    return server
