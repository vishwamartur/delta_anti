
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_system():
    logger.info("Verifying system integration...")
    
    # 1. Verify Config
    import config
    if not hasattr(config, 'MODEL_ACCURACY_CONFIG'):
        logger.error("❌ MODEL_ACCURACY_CONFIG missing from config.py")
        return False
    logger.info("✅ Config verified")
    
    # 2. Verify Model Accuracy Module
    try:
        from analysis.model_accuracy import get_accuracy_tracker
        tracker = get_accuracy_tracker()
        if not tracker:
            logger.error("❌ Failed to initialize accuracy tracker")
            return False
        
        # Test basic functionality
        tracker.record_prediction("test_trade", "lstm", "bullish")
        weight = tracker.get_weight("lstm")
        if weight != 1.0:
            logger.warning(f"⚠️ Unexpected initial weight: {weight}")
            
        tracker.record_outcome("test_trade", "bullish")
        updated_weight = tracker.get_weight("lstm")
        logger.info(f"✅ Model Accuracy Tracker verified (weight: {weight} -> {updated_weight})")
        
    except ImportError as e:
        logger.error(f"❌ Failed to import analysis.model_accuracy: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing tracker: {e}")
        return False
        
    # 3. Verify Signals Module Integration
    try:
        from analysis.signals import signal_generator
        if not hasattr(signal_generator, 'accuracy_tracker'):
            logger.error("❌ SignalGenerator missing accuracy_tracker attribute")
            return False
        logger.info("✅ Signals module integration verified")
    except Exception as e:
        logger.error(f"❌ Error verifying signals module: {e}")
        return False

    # 4. Verify Run System Integration
    try:
        from run_system import IntegratedTradingSystem
        # We can't easily instantiate the full system without connecting to APIs/DBs,
        # but we can check if the class has the methods we added.
        if not hasattr(IntegratedTradingSystem, '_record_trade_outcome'):
             logger.error("❌ IntegratedTradingSystem missing _record_trade_outcome method")
             return False
        logger.info("✅ Run System integration verified")
    except Exception as e:
        logger.error(f"❌ Error verifying run_system: {e}")
        return False

    # 5. Verify Adaptive Strategy Selector
    try:
        if not hasattr(config, 'ADAPTIVE_STRATEGY_CONFIG'):
            logger.error("❌ ADAPTIVE_STRATEGY_CONFIG missing from config.py")
            return False
            
        from strategy.selector import get_strategy_selector, AdaptiveStrategySelector
        selector = get_strategy_selector()
        if not isinstance(selector, AdaptiveStrategySelector):
            logger.error("❌ Failed to initialize AdaptiveStrategySelector")
            return False
            
        logger.info("✅ Adaptive Strategy Selector verified")
    except ImportError as e:
        logger.error(f"❌ Failed to import strategy.selector: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error verifying strategy selector: {e}")
        return False
        
    logger.info("🎉 All systems verified!")
    return True

if __name__ == "__main__":
    success = verify_system()
    sys.exit(0 if success else 1)
