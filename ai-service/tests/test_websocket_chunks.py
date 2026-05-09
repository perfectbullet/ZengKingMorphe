"""
WebSocket client test script for stream chunks real-time updates.

Usage:
    python tests/test_websocket_chunks.py

This script connects to the WebSocket endpoint and displays received chunks.
"""
import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import websockets
except ImportError:
    print("Error: 'websockets' package is not installed.")
    print("Install it with: pip install websockets")
    sys.exit(1)


async def test_websocket_chunks(
    user_id: str = "3",
    employee_id: str = "29",
    session_id: str = "sess_4_3_29",
    host: str = "localhost",
    port: int = 8100,
):
    """
    Test WebSocket connection for real-time stream chunks.

    Args:
        user_id: User ID to filter chunks
        employee_id: Employee ID to filter chunks
        session_id: Session ID to filter chunks
        host: WebSocket server host
        port: WebSocket server port
    """
    uri = f"ws://{host}:{port}/api/chat/ws/view/chunks?user_id={user_id}&employee_id={employee_id}&session_id={session_id}"

    print(f"Connecting to: {uri}")
    print("-" * 60)

    message_count = 0
    chunk_count = 0
    heartbeat_count = 0
    start_time = asyncio.get_event_loop().time()

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Waiting for messages...")
            print("(Press Ctrl+C to disconnect)")
            print("-" * 60)

            while True:
                message = await websocket.recv()
                message_count += 1

                try:
                    data = json.loads(message)

                    if data.get("type") == "heartbeat":
                        heartbeat_count += 1
                        timestamp = data.get("timestamp", "N/A")
                        print(f"[{message_count:04d}] ❤️  Heartbeat - {timestamp}")
                    else:
                        chunk_count += 1
                        chunk_type = data.get("chunk_type", "unknown")
                        sequence = data.get("sequence", 0)
                        chunk_id = data.get("chunk_id", "")[:30]

                        # Different display for different chunk types
                        if chunk_type == "token":
                            content = data.get("chunk_data", {}).get("choices", [{}])[0].get("delta", {}).get("content", "")
                            print(f"[{message_count:04d}] 🔤 Token #{sequence}: '{content}'")
                        elif chunk_type == "done":
                            print(f"[{message_count:04d}] ✅ Done - {chunk_id}")
                        elif chunk_type == "error":
                            print(f"[{message_count:04d}] ❌ Error - {chunk_id}")
                        elif chunk_type == "role":
                            print(f"[{message_count:04d}] 🎭 Role - {chunk_id}")
                        elif chunk_type == "user_query":
                            query = data.get("chunk_data", {}).get("messages", [{}])[-1].get("content", "")[:50]
                            print(f"[{message_count:04d}] 👤 User: {query}...")
                        else:
                            print(f"[{message_count:04d}] 📦 {chunk_type} - {chunk_id}")

                except json.JSONDecodeError:
                    print(f"[{message_count:04d}] ⚠️  Non-JSON message: {message[:100]}")

    except KeyboardInterrupt:
        elapsed = asyncio.get_event_loop().time() - start_time
        print("\n" + "-" * 60)
        print("Disconnected by user")
        print(f"Total messages: {message_count}")
        print(f"  - Chunks: {chunk_count}")
        print(f"  - Heartbeats: {heartbeat_count}")
        print(f"Duration: {elapsed:.1f} seconds")
    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
    except ConnectionRefusedError:
        print(f"\n❌ Connection refused. Is the server running on {host}:{port}?")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


def main():
    """Main entry point for the test script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test WebSocket endpoint for stream chunks"
    )
    parser.add_argument(
        "--user-id",
        default="3",
        help="User ID (default: 3)"
    )
    parser.add_argument(
        "--employee-id",
        default="29",
        help="Employee ID (default: 29)"
    )
    parser.add_argument(
        "--session-id",
        default="sess_4_3_29",
        help="Session ID (default: sess_4_3_29)"
    )
    parser.add_argument(
        "--host",
        default="192.168.8.233",
        help="WebSocket server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="WebSocket server port (default: 8100)"
    )

    args = parser.parse_args()

    asyncio.run(test_websocket_chunks(
        user_id=args.user_id,
        employee_id=args.employee_id,
        session_id=args.session_id,
        host=args.host,
        port=args.port,
    ))


if __name__ == "__main__":
    main()
