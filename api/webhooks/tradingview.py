"""
TradingView Webhook Handler
Receives and processes TradingView alerts for automated trading
"""
import hmac
import hashlib
from datetime import datetime
from typing import Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, Request, HTTPException, Header
from pydantic import BaseModel

from config import config
from strategy.trade_manager import trade_manager


router = APIRouter(prefix="/webhook", tags=["webhooks"])


# Webhook secret for validation (set in .env)
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "your-webhook-secret")


class TradingViewAlert(BaseModel):
    """TradingView alert payload model."""
    action: str  # BUY, SELL, CLOSE
    symbol: str
    price: float
    timestamp: Optional[str] = None
    strategy: Optional[str] = None
    interval: Optional[str] = None
    message: Optional[str] = None
    contracts: Optional[float] = None


class WebhookResponse(BaseModel):
    """Response for webhook processing."""
    status: str
    action: str
    symbol: str
    trade_id: Optional[str] = None
    message: str


def verify_signature(body: bytes, signature: str) -> bool:
    """
    Verify HMAC signature of webhook payload.
    
    Args:
        body: Raw request body
        signature: X-Signature header value
        
    Returns:
        True if signature is valid
    """
    if not signature:
        return False
    
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_signature, signature)


@router.post("/tradingview", response_model=WebhookResponse)
async def tradingview_webhook(
    request: Request,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
):
    """
    Receive and process TradingView alerts.
    
    Expected payload:
    ```json
    {
        "action": "BUY",
        "symbol": "BTCUSD",
        "price": 50000.00,
        "contracts": 1
    }
    ```
    
    Optional: Add X-Signature header with HMAC-SHA256 signature for validation.
    """
    try:
        # Get raw body for signature verification
        body = await request.body()
        
        # Verify signature if provided
        if x_signature and not verify_signature(body, x_signature):
            raise HTTPException(
                status_code=401,
                detail="Invalid webhook signature"
            )
        
        # Parse alert
        data = await request.json()
        alert = TradingViewAlert(**data)
        
        print(f"[Webhook] Received: {alert.action} {alert.symbol} @ {alert.price}")
        
        # Process based on action
        action = alert.action.upper()
        symbol = alert.symbol.upper()
        price = alert.price
        size = alert.contracts or 1
        
        trade_id = None
        message = ""
        
        if action == "BUY":
            # Open long position
            trade = trade_manager.open_trade(
                symbol=symbol,
                direction="LONG",
                entry_price=price,
                size=size,
                signal_type="TRADINGVIEW_BUY"
            )
            if trade:
                trade_id = trade.trade_id
                message = f"Opened LONG position for {symbol}"
            else:
                message = "Failed to open trade (limit reached or error)"
        
        elif action == "SELL":
            # Open short position
            trade = trade_manager.open_trade(
                symbol=symbol,
                direction="SHORT",
                entry_price=price,
                size=size,
                signal_type="TRADINGVIEW_SELL"
            )
            if trade:
                trade_id = trade.trade_id
                message = f"Opened SHORT position for {symbol}"
            else:
                message = "Failed to open trade"
        
        elif action == "CLOSE":
            # Close all positions for symbol
            closed = trade_manager.close_all_for_symbol(symbol, price)
            message = f"Closed {closed} position(s) for {symbol}"
        
        elif action == "CLOSE_LONG":
            # Close long positions only
            closed = trade_manager.close_by_direction(symbol, "LONG", price)
            message = f"Closed {closed} LONG position(s)"
        
        elif action == "CLOSE_SHORT":
            # Close short positions only
            closed = trade_manager.close_by_direction(symbol, "SHORT", price)
            message = f"Closed {closed} SHORT position(s)"
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown action: {action}"
            )
        
        return WebhookResponse(
            status="success",
            action=action,
            symbol=symbol,
            trade_id=trade_id,
            message=message
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Webhook] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telegram")
async def telegram_webhook(request: Request):
    """
    Handle Telegram bot commands for trading.
    
    Commands:
    - /status - Get current positions
    - /signal BTCUSD - Get signal for symbol
    - /buy BTCUSD 1 - Buy 1 contract
    - /sell BTCUSD 1 - Sell 1 contract
    - /close BTCUSD - Close position
    """
    try:
        data = await request.json()
        
        # Parse Telegram message
        message = data.get('message', {})
        text = message.get('text', '')
        chat_id = message.get('chat', {}).get('id')
        
        if not text or not chat_id:
            return {"ok": True}
        
        # Parse command
        parts = text.strip().split()
        command = parts[0].lower()
        
        response_text = ""
        
        if command == '/status':
            trades = trade_manager.get_open_trades()
            if trades:
                response_text = "📊 Open Positions:\n"
                for t in trades:
                    pnl_emoji = "🟢" if t.unrealized_pnl > 0 else "🔴"
                    response_text += f"{pnl_emoji} {t.symbol} {t.direction}: ${t.unrealized_pnl:.2f}\n"
            else:
                response_text = "No open positions"
        
        elif command == '/signal' and len(parts) >= 2:
            symbol = parts[1].upper()
            response_text = f"Signal request for {symbol} (not implemented in webhook)"
        
        elif command == '/stats':
            stats = trade_manager.get_stats()
            response_text = f"📈 Stats:\n"
            response_text += f"Daily P&L: ${stats.get('daily_pnl', 0):.2f}\n"
            response_text += f"Trades Today: {stats.get('trades_today', 0)}\n"
            response_text += f"Win Rate: {stats.get('win_rate', 0):.1f}%"
        
        else:
            response_text = "Unknown command"
        
        return {
            "ok": True,
            "response": response_text
        }
    
    except Exception as e:
        print(f"[Telegram Webhook] Error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/test")
async def test_webhook():
    """Test endpoint to verify webhook is working."""
    return {
        "status": "ok",
        "message": "Webhook endpoint is working",
        "timestamp": datetime.now().isoformat()
    }
