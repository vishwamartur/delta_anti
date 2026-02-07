import sys
import os
import logging

# Add project root to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)

try:
    from strategy.advanced_trade_manager import AdvancedTradeManager, Trade, ExitReason
    print("Import Successful")
    
    # Mock config
    config = {
        'partial_tp_enabled': True,
        'partial_tp_pct': 0.5,
        'time_exit_enabled': True,
        'max_hold_time_minutes': 120
    }
    
    # Init manager
    manager = AdvancedTradeManager(10000, config)
    print("Manager Initialized")
    
    # Check if new attributes exist
    if hasattr(manager, 'partial_tp_enabled') and hasattr(manager, 'time_exit_enabled'):
        print("Param Check: OK")
    else:
        print("Param Check: FAILED")
        
    # Check method existence
    if hasattr(manager, 'execute_partial_exit'):
        print("Method Check: OK")
    else:
        print("Method Check: FAILED")
        
    if ExitReason.PARTIAL_TP:
        print("Enum Check: OK")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
