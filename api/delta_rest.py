"""
Delta Exchange REST API Client with HMAC Authentication
"""
import hashlib
import hmac
import time
import json
import requests
from typing import Optional, Dict, Any, List
import config


class DeltaRestClient:
    """REST API client for Delta Exchange with authentication."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or config.API_KEY
        self.api_secret = api_secret or config.API_SECRET
        self.base_url = config.REST_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'python-delta-trading-bot'
        })
    
    def _generate_signature(self, method: str, path: str, 
                            query_string: str = "", body: str = "") -> tuple:
        """Generate HMAC signature for authenticated requests."""
        timestamp = str(int(time.time()))
        signature_data = method + timestamp + path + query_string + body
        
        signature = hmac.new(
            bytes(self.api_secret, 'utf-8'),
            bytes(signature_data, 'utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature, timestamp
    
    def _request(self, method: str, path: str, 
                 params: Dict = None, data: Dict = None, 
                 authenticated: bool = False) -> Dict:
        """Make HTTP request to Delta Exchange API."""
        url = f"{self.base_url}{path}"
        
        query_string = ""
        if params:
            query_string = "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        
        body = ""
        if data:
            body = json.dumps(data)
        
        headers = {}
        if authenticated:
            signature, timestamp = self._generate_signature(
                method, path, query_string, body
            )
            headers = {
                'api-key': self.api_key,
                'signature': signature,
                'timestamp': timestamp
            }
        
        try:
            response = self.session.request(
                method=method,
                url=url + query_string,
                data=body if body else None,
                headers=headers,
                timeout=(3, 30)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            return {"error": str(e)}
    
    # ========== Public Endpoints ==========
    
    def get_products(self, contract_types: str = None) -> Dict:
        """Get list of all trading products."""
        params = {}
        if contract_types:
            params['contract_types'] = contract_types
        return self._request('GET', '/v2/products', params=params)
    
    def get_product(self, symbol: str) -> Dict:
        """Get product details by symbol."""
        return self._request('GET', f'/v2/products/{symbol}')
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker for a specific product."""
        return self._request('GET', f'/v2/tickers/{symbol}')
    
    def get_tickers(self, contract_types: str = None) -> Dict:
        """Get tickers for all products."""
        params = {}
        if contract_types:
            params['contract_types'] = contract_types
        return self._request('GET', '/v2/tickers', params=params)
    
    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """Get L2 orderbook for a symbol."""
        params = {'depth': depth}
        return self._request('GET', f'/v2/l2orderbook/{symbol}', params=params)
    
    def get_candles(self, symbol: str, resolution: str, 
                    start: int, end: int) -> Dict:
        """
        Get historical OHLC candles.
        
        Args:
            symbol: Product symbol (e.g., 'BTCUSD')
            resolution: Candle resolution ('1m', '5m', '15m', '1h', '4h', '1d')
            start: Start timestamp (Unix seconds)
            end: End timestamp (Unix seconds)
        """
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'start': start,
            'end': end
        }
        return self._request('GET', '/v2/history/candles', params=params)
    
    def get_recent_trades(self, symbol: str) -> Dict:
        """Get recent public trades for a symbol."""
        return self._request('GET', f'/v2/trades/{symbol}')
    
    # ========== Authenticated Endpoints ==========
    
    def get_wallet_balances(self) -> Dict:
        """Get wallet balances."""
        return self._request('GET', '/v2/wallet/balances', authenticated=True)
    
    def get_positions(self) -> Dict:
        """Get all open positions."""
        return self._request('GET', '/v2/positions/margined', authenticated=True)
    
    def get_position(self, symbol: str) -> Dict:
        """Get position for a specific symbol."""
        params = {'product_symbol': symbol}
        return self._request('GET', '/v2/positions', params=params, authenticated=True)
    
    def get_active_orders(self, symbol: str = None, state: str = "open") -> Dict:
        """Get active orders."""
        params = {'state': state}
        if symbol:
            params['product_symbol'] = symbol
        return self._request('GET', '/v2/orders', params=params, authenticated=True)
    
    def place_order(self, symbol: str, side: str, size: int, 
                    order_type: str = "market_order",
                    limit_price: str = None,
                    stop_price: str = None,
                    stop_order_type: str = None,
                    time_in_force: str = "gtc",
                    reduce_only: bool = False,
                    client_order_id: str = None) -> Dict:
        """
        Place a new order.
        
        Args:
            symbol: Product symbol
            side: 'buy' or 'sell'
            size: Order size in contracts
            order_type: 'limit_order' or 'market_order'
            limit_price: Limit price (required for limit orders)
            stop_price: Stop trigger price
            stop_order_type: 'stop_loss_order' or 'take_profit_order'
            time_in_force: 'gtc', 'ioc', 'fok'
            reduce_only: If True, order can only reduce position
            client_order_id: Optional custom order ID
        """
        data = {
            "product_symbol": symbol,
            "side": side,
            "size": size,
            "order_type": order_type,
            "time_in_force": time_in_force,
            "reduce_only": reduce_only
        }
        
        if limit_price:
            data["limit_price"] = str(limit_price)
        if stop_price:
            data["stop_price"] = str(stop_price)
        if stop_order_type:
            data["stop_order_type"] = stop_order_type
        if client_order_id:
            data["client_order_id"] = client_order_id
            
        return self._request('POST', '/v2/orders', data=data, authenticated=True)
    
    def cancel_order(self, order_id: int = None, 
                     client_order_id: str = None) -> Dict:
        """Cancel an order by ID."""
        data = {}
        if order_id:
            data["id"] = order_id
        if client_order_id:
            data["client_order_id"] = client_order_id
        return self._request('DELETE', '/v2/orders', data=data, authenticated=True)
    
    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """Cancel all open orders."""
        data = {}
        if symbol:
            data["product_symbol"] = symbol
        return self._request('DELETE', '/v2/orders/all', data=data, authenticated=True)
    
    def get_order_history(self, symbol: str = None, 
                          start_time: int = None, 
                          end_time: int = None,
                          page_size: int = 100) -> Dict:
        """Get order history."""
        params = {'page_size': page_size}
        if symbol:
            params['product_symbol'] = symbol
        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time
        return self._request('GET', '/v2/orders/history', params=params, authenticated=True)
    
    def get_fills(self, symbol: str = None, 
                  start_time: int = None,
                  end_time: int = None,
                  page_size: int = 100) -> Dict:
        """Get trade fills history."""
        params = {'page_size': page_size}
        if symbol:
            params['product_symbol'] = symbol
        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time
        return self._request('GET', '/v2/fills', params=params, authenticated=True)
    
    # ========== Leverage Management ==========
    
    def set_leverage(self, product_id: int, leverage: int = None) -> Dict:
        """
        Set leverage for a product.
        
        Args:
            product_id: The product ID (not symbol)
            leverage: Leverage amount (default: uses DEFAULT_LEVERAGE from config, which is 200x)
        
        Returns:
            API response with leverage, order_margin, and product_id
        """
        if leverage is None:
            leverage = config.DEFAULT_LEVERAGE
        
        data = {"leverage": leverage}
        return self._request(
            'POST', 
            f'/v2/products/{product_id}/orders/leverage', 
            data=data, 
            authenticated=True
        )
    
    def get_leverage(self, product_id: int) -> Dict:
        """
        Get current leverage for a product.
        
        Args:
            product_id: The product ID (not symbol)
        
        Returns:
            API response with current leverage settings
        """
        return self._request(
            'GET', 
            f'/v2/products/{product_id}/orders/leverage', 
            authenticated=True
        )
    
    def get_product_id(self, symbol: str) -> Optional[int]:
        """
        Get product ID from symbol.
        
        Args:
            symbol: Product symbol (e.g., 'BTCUSD')
            
        Returns:
            Product ID or None if not found
        """
        result = self.get_product(symbol)
        if 'result' in result and result['result']:
            return result['result'].get('id')
        return None
    
    def set_leverage_by_symbol(self, symbol: str, leverage: int = None) -> Dict:
        """
        Convenience method to set leverage using symbol instead of product ID.
        
        Args:
            symbol: Product symbol (e.g., 'BTCUSD')
            leverage: Leverage amount (default: 200x from config)
            
        Returns:
            API response or error
        """
        product_id = self.get_product_id(symbol)
        if product_id is None:
            return {"error": f"Could not find product ID for symbol: {symbol}"}
        return self.set_leverage(product_id, leverage)
    
    # ========== Auto Topup (Liquidation Prevention) ==========
    
    def set_auto_topup(self, product_id: int, enabled: bool = True) -> Dict:
        """
        Enable or disable auto topup for a position.
        When enabled, margin is automatically added to prevent liquidation.
        
        Args:
            product_id: The product ID
            enabled: True to enable auto topup (default), False to disable
            
        Returns:
            API response with position details
        """
        data = {
            "product_id": product_id,
            "auto_topup": "true" if enabled else "false"
        }
        return self._request(
            'PUT',
            '/v2/positions/auto_topup',
            data=data,
            authenticated=True
        )
    
    def set_auto_topup_by_symbol(self, symbol: str, enabled: bool = True) -> Dict:
        """
        Enable or disable auto topup using symbol.
        
        Args:
            symbol: Product symbol (e.g., 'BTCUSD')
            enabled: True to enable auto topup (default), False to disable
            
        Returns:
            API response or error
        """
        product_id = self.get_product_id(symbol)
        if product_id is None:
            return {"error": f"Could not find product ID for symbol: {symbol}"}
        return self.set_auto_topup(product_id, enabled)
    
    def enable_auto_topup_for_all_symbols(self) -> Dict[str, Dict]:
        """
        Enable auto topup for all configured trading symbols.
        This is useful for preventing liquidation on all positions at once.
        
        Returns:
            Dict mapping symbol to API response
        """
        results = {}
        for symbol in config.TRADING_SYMBOLS:
            results[symbol] = self.set_auto_topup_by_symbol(symbol, enabled=True)
        return results


# Singleton instance
rest_client = DeltaRestClient()
