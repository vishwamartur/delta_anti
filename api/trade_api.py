"""
API endpoints for Trade Manager Dashboard
REST API for managing trades, viewing statistics, and controlling positions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from strategy.advanced_trade_manager import (
    get_trade_manager, ExitReason, TradeDirection
)


router = APIRouter(prefix="/api/trades", tags=["trades"])


class TradeResponse(BaseModel):
    """Trade response model"""
    trade_id: str
    symbol: str
    direction: str
    entry_price: Optional[float]
    entry_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    state: str
    current_price: float
    unrealized_pnl: float
    pnl_percent: float
    duration_minutes: int


class StatsResponse(BaseModel):
    """Statistics response model"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    open_positions: int
    total_pnl: float
    daily_pnl: float
    account_balance: float
    max_drawdown: float


class CloseTradeRequest(BaseModel):
    """Request to close a trade"""
    exit_price: Optional[float] = None
    reason: str = "MANUAL"


@router.get("/active", response_model=Dict)
async def get_active_trades():
    """Get all active trades"""
    tm = get_trade_manager()
    if not tm:
        raise HTTPException(status_code=503, detail="Trade manager not initialized")
    
    trades = [t.to_dict() for t in tm.open_trades.values()]
    return {
        "trades": trades,
        "count": len(trades)
    }


@router.get("/history", response_model=Dict)
async def get_trade_history(limit: int = 20):
    """Get trade history"""
    tm = get_trade_manager()
    if not tm:
        raise HTTPException(status_code=503, detail="Trade manager not initialized")
    
    trades = [t.to_dict() for t in tm.trade_history[:limit]]
    return {
        "trades": trades,
        "count": len(tm.trade_history)
    }


@router.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get trading statistics"""
    tm = get_trade_manager()
    if not tm:
        raise HTTPException(status_code=503, detail="Trade manager not initialized")
    
    return tm.get_statistics()


@router.get("/{trade_id}", response_model=Dict)
async def get_trade(trade_id: str):
    """Get specific trade by ID"""
    tm = get_trade_manager()
    if not tm:
        raise HTTPException(status_code=503, detail="Trade manager not initialized")
    
    trade = tm.trades.get(trade_id)
    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
    
    return trade.to_dict()


@router.post("/{trade_id}/close", response_model=Dict)
async def close_trade(trade_id: str, request: CloseTradeRequest = None):
    """Manually close a trade"""
    tm = get_trade_manager()
    if not tm:
        raise HTTPException(status_code=503, detail="Trade manager not initialized")
    
    # Parse exit reason
    reason = ExitReason.MANUAL
    if request and request.reason:
        try:
            reason = ExitReason[request.reason.upper()]
        except KeyError:
            reason = ExitReason.MANUAL
    
    exit_price = request.exit_price if request else None
    
    trade = tm.close_trade(trade_id, reason, exit_price)
    
    if trade:
        return {
            "status": "success",
            "message": f"Trade {trade_id} closed",
            "realized_pnl": trade.realized_pnl,
            "pnl_percent": trade.pnl_percent
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to close trade")


@router.post("/close-all/{symbol}", response_model=Dict)
async def close_all_trades(symbol: str, exit_price: float = None):
    """Close all trades for a symbol"""
    tm = get_trade_manager()
    if not tm:
        raise HTTPException(status_code=503, detail="Trade manager not initialized")
    
    closed_count = tm.close_all_for_symbol(symbol.upper(), exit_price or 0)
    
    return {
        "status": "success",
        "message": f"Closed {closed_count} trades for {symbol}",
        "closed_count": closed_count
    }


@router.get("/config/risk", response_model=Dict)
async def get_risk_config():
    """Get current risk configuration"""
    tm = get_trade_manager()
    if not tm:
        raise HTTPException(status_code=503, detail="Trade manager not initialized")
    
    rm = tm.risk_manager
    return {
        "account_balance": rm.account_balance,
        "max_risk_per_trade": rm.max_risk_per_trade,
        "max_positions": rm.max_positions,
        "max_daily_loss": rm.max_daily_loss,
        "max_drawdown": rm.max_drawdown,
        "daily_pnl": rm.daily_pnl,
        "current_drawdown": rm.current_drawdown
    }
