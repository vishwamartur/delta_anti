"""
Delta Exchange WebSocket Client for Real-Time Data
"""
import hashlib
import hmac
import json
import time
import threading
from typing import Callable, Dict, List, Optional
import websocket
import config


class DeltaWebSocketClient:
    """WebSocket client for Delta Exchange real-time data feeds."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or config.API_KEY
        self.api_secret = api_secret or config.API_SECRET
        self.ws_url = config.WS_URL
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self.is_connected = False
        self.is_authenticated = False
        self.reconnect_count = 0
        self.max_reconnects = 5
        
        # Callback handlers
        self.on_ticker: Optional[Callable] = None
        self.on_candlestick: Optional[Callable] = None
        self.on_orderbook: Optional[Callable] = None
        self.on_trades: Optional[Callable] = None
        self.on_mark_price: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Subscriptions to restore on reconnect
        self._subscriptions: List[Dict] = []
        
        # Thread for running WebSocket
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def _generate_signature(self) -> tuple:
        """Generate signature for WebSocket authentication."""
        method = 'GET'
        timestamp = str(int(time.time()))
        path = '/live'
        signature_data = method + timestamp + path
        
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(signature_data, 'utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature, timestamp
    
    def _on_open(self, ws):
        """Handle WebSocket connection open."""
        print("[WS] Connected to Delta Exchange")
        self.is_connected = True
        self.reconnect_count = 0
        
        # Authenticate if we have API keys
        if self.api_key and self.api_secret:
            self._authenticate()
        
        # Restore subscriptions
        for sub in self._subscriptions:
            self._send_subscribe(sub['channel'], sub['symbols'])
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        print(f"[WS] Connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        self.is_authenticated = False
        
        # Attempt reconnection
        if not self._stop_event.is_set() and self.reconnect_count < self.max_reconnects:
            self.reconnect_count += 1
            print(f"[WS] Attempting reconnect ({self.reconnect_count}/{self.max_reconnects})...")
            time.sleep(2 ** self.reconnect_count)  # Exponential backoff
            self._connect()
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        print(f"[WS] Error: {error}")
        if self.on_error:
            self.on_error(error)
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            # Handle authentication response
            if msg_type == 'key-auth':
                if data.get('success'):
                    print("[WS] Authentication successful")
                    self.is_authenticated = True
                    # Subscribe to private channels
                    self._subscribe_private_channels()
                else:
                    print(f"[WS] Authentication failed: {data}")
                return
            
            # Handle subscription confirmation
            if msg_type == 'subscriptions':
                print(f"[WS] Subscribed to: {data.get('channels', [])}")
                return
            
            # Route message to appropriate handler
            self._route_message(data)
            
        except json.JSONDecodeError as e:
            print(f"[WS] Failed to parse message: {e}")
    
    def _route_message(self, data: Dict):
        """Route message to appropriate callback handler."""
        msg_type = data.get('type', '')
        
        # Ticker updates
        if msg_type == 'v2/ticker' and self.on_ticker:
            self.on_ticker(data)
        
        # Candlestick updates
        elif msg_type.startswith('candlestick_') and self.on_candlestick:
            self.on_candlestick(data)
        
        # Orderbook updates
        elif msg_type in ['l2_orderbook', 'l2_updates'] and self.on_orderbook:
            self.on_orderbook(data)
        
        # Trade updates
        elif msg_type == 'all_trades' and self.on_trades:
            self.on_trades(data)
        
        # Mark price updates
        elif msg_type == 'mark_price' and self.on_mark_price:
            self.on_mark_price(data)
        
        # Order updates (private)
        elif msg_type == 'orders' and self.on_order_update:
            self.on_order_update(data)
        
        # Position updates (private)
        elif msg_type == 'positions' and self.on_position_update:
            self.on_position_update(data)
    
    def _authenticate(self):
        """Send authentication message."""
        signature, timestamp = self._generate_signature()
        auth_payload = {
            "type": "key-auth",
            "payload": {
                "api-key": self.api_key,
                "signature": signature,
                "timestamp": timestamp
            }
        }
        self.ws.send(json.dumps(auth_payload))
    
    def _subscribe_private_channels(self):
        """Subscribe to private channels after authentication."""
        # Subscribe to orders channel
        self._send_subscribe("orders", ["all"])
        # Subscribe to positions channel  
        self._send_subscribe("positions", ["all"])
    
    def _send_subscribe(self, channel: str, symbols: List[str]):
        """Send subscription request."""
        if not self.is_connected:
            return
            
        payload = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {
                        "name": channel,
                        "symbols": symbols
                    }
                ]
            }
        }
        self.ws.send(json.dumps(payload))
    
    def _send_unsubscribe(self, channel: str, symbols: List[str]):
        """Send unsubscription request."""
        if not self.is_connected:
            return
            
        payload = {
            "type": "unsubscribe",
            "payload": {
                "channels": [
                    {
                        "name": channel,
                        "symbols": symbols
                    }
                ]
            }
        }
        self.ws.send(json.dumps(payload))
    
    def _connect(self):
        """Create and connect WebSocket."""
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.ws.run_forever()
    
    # ========== Public API ==========
    
    def start(self):
        """Start WebSocket connection in background thread."""
        if self._thread and self._thread.is_alive():
            print("[WS] Already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._connect, daemon=True)
        self._thread.start()
        
        # Wait for connection
        timeout = 10
        start = time.time()
        while not self.is_connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        if not self.is_connected:
            print("[WS] Connection timeout")
    
    def stop(self):
        """Stop WebSocket connection."""
        self._stop_event.set()
        if self.ws:
            self.ws.close()
        if self._thread:
            self._thread.join(timeout=5)
        self.is_connected = False
        self.is_authenticated = False
        print("[WS] Disconnected")
    
    def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates for symbols."""
        channel = "v2/ticker"
        self._subscriptions.append({"channel": channel, "symbols": symbols})
        self._send_subscribe(channel, symbols)
    
    def subscribe_candlestick(self, symbols: List[str], resolution: str = "1m"):
        """Subscribe to candlestick updates."""
        channel = f"candlestick_{resolution}"
        self._subscriptions.append({"channel": channel, "symbols": symbols})
        self._send_subscribe(channel, symbols)
    
    def subscribe_orderbook(self, symbols: List[str]):
        """Subscribe to L2 orderbook updates."""
        channel = "l2_orderbook"
        self._subscriptions.append({"channel": channel, "symbols": symbols})
        self._send_subscribe(channel, symbols)
    
    def subscribe_trades(self, symbols: List[str]):
        """Subscribe to public trade updates."""
        channel = "all_trades"
        self._subscriptions.append({"channel": channel, "symbols": symbols})
        self._send_subscribe(channel, symbols)
    
    def subscribe_mark_price(self, symbols: List[str]):
        """Subscribe to mark price updates."""
        channel = "mark_price"
        # Mark price symbols need MARK: prefix
        mark_symbols = [f"MARK:{s}" if not s.startswith("MARK:") else s for s in symbols]
        self._subscriptions.append({"channel": channel, "symbols": mark_symbols})
        self._send_subscribe(channel, mark_symbols)
    
    def unsubscribe(self, channel: str, symbols: List[str]):
        """Unsubscribe from a channel."""
        self._subscriptions = [
            s for s in self._subscriptions 
            if not (s['channel'] == channel and s['symbols'] == symbols)
        ]
        self._send_unsubscribe(channel, symbols)


# Singleton instance
ws_client = DeltaWebSocketClient()
