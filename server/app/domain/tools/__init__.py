"""High-level tools exposed to business Agents.

The MCP implementation and external API details stay behind these small
interfaces.  Agents receive only the tools listed in their manifest.
"""

from app.domain.tools.commerce_mcp import (
    ALLOWED_COMMERCE_PLATFORMS,
    CommerceMcpTransport,
    CommerceProductDetailTool,
    CommerceResearchTool,
    CommerceReviewsTool,
    HttpCommerceMcpTransport,
    UnavailableCommerceTransport,
)
from app.domain.tools.web_search import WebSearchTool
from app.domain.tools.local import ImageUnderstandingTool, ProductDetailTool

__all__ = [
    "ALLOWED_COMMERCE_PLATFORMS",
    "CommerceMcpTransport",
    "CommerceProductDetailTool",
    "CommerceResearchTool",
    "CommerceReviewsTool",
    "HttpCommerceMcpTransport",
    "UnavailableCommerceTransport",
    "WebSearchTool",
    "ImageUnderstandingTool",
    "ProductDetailTool",
]
