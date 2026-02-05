
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from analysis.smart_money import SmartMoneyAnalyzer

def create_smc_pattern_data():
    """Create data with specific SMC patterns."""
    dates = [datetime.now() - timedelta(hours=i) for i in range(100)]
    dates.reverse()
    
    # Base price path
    prices = np.linspace(50000, 50000, 100) 
    
    data = []
    
    # 1. Create a Bullish Order Block setup
    # Candle 0: Bearish candle (The OB)
    # Candle 1: Strong Bullish Impulse
    # Candle 2: Gap up (FVG)
    
    for i in range(100):
        ts = dates[i]
        
        if i == 50: 
            # Bullish OB (Bearish candle before move)
            open_p, close_p = 50000, 49900
            high_p, low_p = 50010, 49890
        elif i == 51:
            # Strong Impulse
            open_p, close_p = 49900, 50200 # Breaker
            high_p, low_p = 50250, 49890
        elif i == 52:
            # Continuation (leaving FVG between 50 high and 52 low)
            open_p, close_p = 50200, 50300
            high_p, low_p = 50350, 50200 # Low > 50's High (50010) -> Gap
        else:
            # Random noise
            base = 50000 + (i-50)*10 if i > 50 else 50000 + np.random.randn()*50
            open_p = base
            close_p = base + np.random.randn()*20
            high_p = max(open_p, close_p) + 10
            low_p = min(open_p, close_p) - 10
            
        data.append({
            'timestamp': ts,
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': 1000
        })
        
    return pd.DataFrame(data).set_index('timestamp')

def main():
    print("Testing Smart Money Concepts Analysis...")
    
    df = create_smc_pattern_data()
    analyzer = SmartMoneyAnalyzer()
    
    result = analyzer.analyze(df)
    
    print(f"\nAnalysis Results:")
    print(f"Structure: {result.market_structure}")
    
    print(f"\nOrder Blocks Found: {len(result.order_blocks)}")
    for ob in result.order_blocks:
        print(f"- {ob.direction.upper()} OB at {ob.price_low:.2f}-{ob.price_high:.2f} (Str: {ob.strength})")
        
    print(f"\nFair Value Gaps Found: {len(result.fvgs)}")
    for fvg in result.fvgs:
        print(f"- {fvg.direction.upper()} FVG at {fvg.price_low:.2f}-{fvg.price_high:.2f}")
        
    print(f"\nLiquidity Sweeps: {len(result.liquidity_sweeps)}")
    for sweep in result.liquidity_sweeps:
        print(f"- {sweep.direction.upper()} Sweep at {sweep.price_high:.2f}")
        
    if len(result.order_blocks) > 0 or len(result.fvgs) > 0:
        print("\n✅ Verification Successful: SMC patterns detected.")
    else:
        print("\n❌ Verification Failed: No patterns detected.")

if __name__ == "__main__":
    main()
