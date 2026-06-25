#!/usr/bin/env bash
# Start the courier OCR server and expose it via Cloudflare Tunnel
set -e

cd "$(dirname "$0")"

# Start FastAPI in the background
echo "Starting server..."
python main.py &
SERVER_PID=$!

# Give the server a moment to bind
sleep 2

# Start cloudflared — prints a public https URL to the console
echo ""
echo "Starting Cloudflare Tunnel..."
cloudflared tunnel --url http://localhost:8000 &
TUNNEL_PID=$!

echo ""
echo "Press Ctrl+C to stop everything."

# Wait and clean up on exit
trap "kill $SERVER_PID $TUNNEL_PID 2>/dev/null; exit" INT TERM
wait
