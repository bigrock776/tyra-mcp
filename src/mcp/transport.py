"""
Transport layer implementations for MCP server.

This module provides different transport mechanisms for the MCP server:
- Standard I/O (stdio) for direct process communication
- Server-Sent Events (SSE) for HTTP-based communication
- WebSocket support for real-time bidirectional communication
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""

    @abstractmethod
    async def start(self, message_handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Start the transport and begin handling messages."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport and cleanup resources."""
        pass

    @abstractmethod
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message through the transport."""
        pass


class StdioTransport(MCPTransport):
    """
    Standard I/O transport for MCP.

    This is the primary transport used by MCP for direct process communication.
    Messages are exchanged via stdin/stdout as JSON-RPC 2.0 messages.
    """

    def __init__(self):
        self.message_handler: Optional[Callable] = None
        self._running = False
        self._reader_task: Optional[asyncio.Task] = None

    async def start(self, message_handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Start stdio transport."""
        self.message_handler = message_handler
        self._running = True

        # Start reading from stdin
        self._reader_task = asyncio.create_task(self._read_stdin())

        logger.info("Stdio transport started")

    async def stop(self) -> None:
        """Stop stdio transport."""
        self._running = False

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        logger.info("Stdio transport stopped")

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send message to stdout."""
        try:
            json_message = json.dumps(message)
            print(json_message, flush=True)

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    async def _read_stdin(self) -> None:
        """Read messages from stdin."""
        try:
            while self._running:
                # Read line from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, input)

                if not line:
                    continue

                try:
                    # Parse JSON message
                    message = json.loads(line.strip())

                    # Handle message
                    if self.message_handler:
                        response = await self.message_handler(message)
                        if response:
                            await self.send_message(response)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    await self.send_message(
                        {
                            "jsonrpc": "2.0",
                            "error": {"code": -32700, "message": "Parse error"},
                        }
                    )

                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self.send_message(
                        {
                            "jsonrpc": "2.0",
                            "error": {"code": -32603, "message": "Internal error"},
                        }
                    )

        except asyncio.CancelledError:
            logger.debug("Stdin reader cancelled")
        except Exception as e:
            logger.error(f"Stdin reader error: {e}")


class SSETransport(MCPTransport):
    """
    Server-Sent Events transport for MCP.

    Provides HTTP-based communication using SSE for server-to-client messages
    and HTTP POST for client-to-server messages.
    """

    def __init__(self, host: str = "localhost", port: int = 3000):
        self.host = host
        self.port = port
        self.message_handler: Optional[Callable] = None
        self._running = False
        self._server: Optional[Any] = None
        self._clients: set = set()

    async def start(self, message_handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Start SSE transport with HTTP server."""
        try:
            from aiohttp import web, web_request
            from aiohttp.web import Response

            self.message_handler = message_handler
            self._running = True

            app = web.Application()

            # SSE endpoint for server-to-client messages
            app.router.add_get("/events", self._handle_sse)

            # HTTP endpoint for client-to-server messages
            app.router.add_post("/message", self._handle_http_message)

            # Health check endpoint
            app.router.add_get("/health", self._handle_health)

            # Start server
            runner = web.AppRunner(app)
            await runner.setup()

            site = web.TCPSite(runner, self.host, self.port)
            await site.start()

            self._server = runner

            logger.info(f"SSE transport started on {self.host}:{self.port}")

        except ImportError:
            raise RuntimeError("aiohttp is required for SSE transport")

    async def stop(self) -> None:
        """Stop SSE transport."""
        self._running = False

        if self._server:
            await self._server.cleanup()

        logger.info("SSE transport stopped")

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send message to all SSE clients."""
        if not self._clients:
            return

        try:
            json_message = json.dumps(message)

            # Send to all connected clients
            disconnected_clients = set()
            for client in self._clients:
                try:
                    await client.write(f"data: {json_message}\n\n".encode())
                except Exception:
                    disconnected_clients.add(client)

            # Remove disconnected clients
            self._clients -= disconnected_clients

        except Exception as e:
            logger.error(f"Failed to send SSE message: {e}")

    async def _handle_sse(self, request) -> Any:
        """Handle SSE connection."""
        from aiohttp.web import StreamResponse

        response = StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["Access-Control-Allow-Origin"] = "*"

        await response.prepare(request)

        # Add client to active connections
        self._clients.add(response)

        try:
            # Send initial connection message
            await response.write(b'data: {"type": "connected"}\n\n')

            # Keep connection alive
            while self._running:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                await response.write(b'data: {"type": "heartbeat"}\n\n')

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"SSE connection error: {e}")
        finally:
            self._clients.discard(response)

        return response

    async def _handle_http_message(self, request) -> Any:
        """Handle HTTP message from client."""
        from aiohttp.web import Response

        try:
            # Parse JSON message
            message = await request.json()

            # Handle message
            if self.message_handler:
                response_data = await self.message_handler(message)
                if response_data:
                    return Response(
                        text=json.dumps(response_data), content_type="application/json"
                    )

            return Response(status=200)

        except Exception as e:
            logger.error(f"Error handling HTTP message: {e}")
            return Response(
                text=json.dumps(
                    {"error": {"code": -32603, "message": "Internal error"}}
                ),
                content_type="application/json",
                status=500,
            )

    async def _handle_health(self, request) -> Any:
        """Handle health check."""
        from aiohttp.web import Response

        return Response(
            text=json.dumps(
                {"status": "healthy", "transport": "sse", "clients": len(self._clients)}
            ),
            content_type="application/json",
        )


class WebSocketTransport(MCPTransport):
    """
    WebSocket transport for MCP.

    Provides real-time bidirectional communication using WebSockets.
    """

    def __init__(self, host: str = "localhost", port: int = 3001):
        self.host = host
        self.port = port
        self.message_handler: Optional[Callable] = None
        self._running = False
        self._server: Optional[Any] = None
        self._clients: set = set()

    async def start(self, message_handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Start WebSocket transport."""
        try:
            import websockets
            from websockets.server import serve

            self.message_handler = message_handler
            self._running = True

            # Start WebSocket server
            self._server = await serve(self._handle_websocket, self.host, self.port)

            logger.info(f"WebSocket transport started on {self.host}:{self.port}")

        except ImportError:
            raise RuntimeError("websockets is required for WebSocket transport")

    async def stop(self) -> None:
        """Stop WebSocket transport."""
        self._running = False

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("WebSocket transport stopped")

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send message to all WebSocket clients."""
        if not self._clients:
            return

        try:
            json_message = json.dumps(message)

            # Send to all connected clients
            disconnected_clients = set()
            for client in self._clients:
                try:
                    await client.send(json_message)
                except Exception:
                    disconnected_clients.add(client)

            # Remove disconnected clients
            self._clients -= disconnected_clients

        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

    async def _handle_websocket(self, websocket, path) -> None:
        """Handle WebSocket connection."""
        # Add client to active connections
        self._clients.add(websocket)

        try:
            # Send initial connection message
            await websocket.send(json.dumps({"type": "connected"}))

            # Handle incoming messages
            async for message in websocket:
                try:
                    # Parse JSON message
                    data = json.loads(message)

                    # Handle message
                    if self.message_handler:
                        response = await self.message_handler(data)
                        if response:
                            await websocket.send(json.dumps(response))

                except json.JSONDecodeError as e:
                    await websocket.send(
                        json.dumps(
                            {"error": {"code": -32700, "message": "Parse error"}}
                        )
                    )

                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    await websocket.send(
                        json.dumps(
                            {"error": {"code": -32603, "message": "Internal error"}}
                        )
                    )

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self._clients.discard(websocket)


def create_transport(transport_type: str = "stdio", **kwargs) -> MCPTransport:
    """
    Create a transport instance based on type.

    Args:
        transport_type: Type of transport ("stdio", "sse", "websocket")
        **kwargs: Additional transport-specific arguments

    Returns:
        MCPTransport instance

    Raises:
        ValueError: If transport type is unknown
    """
    transport_type = transport_type.lower()

    if transport_type == "stdio":
        return StdioTransport()
    elif transport_type == "sse":
        return SSETransport(**kwargs)
    elif transport_type == "websocket":
        return WebSocketTransport(**kwargs)
    else:
        raise ValueError(f"Unknown transport type: {transport_type}")


async def run_with_transport(
    server_instance, transport_type: str = "stdio", **transport_kwargs
) -> None:
    """
    Run MCP server with specified transport.

    Args:
        server_instance: MCP server instance
        transport_type: Type of transport to use
        **transport_kwargs: Transport-specific arguments
    """
    transport = create_transport(transport_type, **transport_kwargs)

    try:
        # Define message handler
        async def handle_message(message: Dict[str, Any]) -> Dict[str, Any]:
            # Route message to server instance
            # This would need to be implemented based on the server's message handling
            return await server_instance.handle_message(message)

        # Start transport
        await transport.start(handle_message)

        # Keep running until stopped
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Transport stopped by user")
    except Exception as e:
        logger.error(f"Transport error: {e}")
        raise
    finally:
        await transport.stop()
