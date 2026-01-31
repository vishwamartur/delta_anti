"""
Launch the complete Delta Anti Trading System
Runs API server + Trading system with Advanced Trade Manager
"""
import threading
import logging
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from main import TradingSystem
from api.server.main import run_server
from strategy.advanced_trade_manager import initialize_trade_manager
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """Run the complete trading system with API server."""
    
    print("=" * 70)
    print("DELTA ANTI TRADING SYSTEM - FULL STACK")
    print("=" * 70)
    
    # Initialize trade manager
    tm = initialize_trade_manager(
        account_balance=config.INITIAL_ACCOUNT_BALANCE,
        trade_config=config.TRADE_MANAGER_CONFIG
    )
    print(f"[+] Trade Manager initialized: ${config.INITIAL_ACCOUNT_BALANCE:,.2f}")
    print(f"    - Auto Execution: {config.TRADE_MANAGER_CONFIG.get('enable_auto_execution', False)}")
    print(f"    - Max Positions: {config.TRADE_MANAGER_CONFIG.get('max_positions', 5)}")
    print(f"    - Risk Per Trade: {config.TRADE_MANAGER_CONFIG.get('max_risk_per_trade', 0.02) * 100}%")
    
    # Start API server in thread
    api_thread = threading.Thread(
        target=run_server,
        kwargs={'host': '0.0.0.0', 'port': 8000},
        daemon=True
    )
    api_thread.start()
    
    print("[+] API Server starting on http://localhost:8000")
    print("    - Docs: http://localhost:8000/docs")
    print("    - Trade API: http://localhost:8000/api/trades/active")
    
    # Wait for API server to start
    time.sleep(2)
    
    print("=" * 70)
    print("[+] Starting Trading System...")
    print("=" * 70)
    
    # Start main trading system
    system = TradingSystem()
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\n[!] Shutdown requested...")
        system.stop()
        print("[+] System stopped cleanly")


if __name__ == "__main__":
    main()
