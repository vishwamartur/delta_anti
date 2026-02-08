"""
Real-Time Trading Dashboard
Displays market data, indicators, signals, and positions.
"""
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

from analysis.signals import TradeSignal, SignalType, SignalStrength
from analysis.indicators import IndicatorValues
from strategy.trade_manager import Trade, TradeStatus
import config


class Dashboard:
    """
    Real-time console dashboard for trading system.
    Uses the 'rich' library for beautiful terminal output.
    """
    
    def __init__(self):
        self.console = Console()
        self.last_update = datetime.now()
        self._start_time = datetime.now()  # Track uptime
        
        # Store latest data
        self._prices: Dict[str, float] = {}
        self._indicators: Dict[str, IndicatorValues] = {}
        self._signals: Dict[str, TradeSignal] = {}
        self._trades: List[Trade] = []
        self._stats: Dict = {}
        self._messages: List[str] = []
    
    def _get_signal_color(self, signal: TradeSignal) -> str:
        """Get color for signal type."""
        if signal.signal_type == SignalType.LONG:
            return "green"
        elif signal.signal_type == SignalType.SHORT:
            return "red"
        elif signal.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]:
            return "yellow"
        return "white"
    
    def _get_pnl_color(self, pnl: float) -> str:
        """Get color for P&L value."""
        if pnl > 0:
            return "green"
        elif pnl < 0:
            return "red"
        return "white"
    
    def _create_header(self) -> Panel:
        """Create header panel."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = Text()
        header_text.append("DELTA EXCHANGE TRADING SYSTEM", style="bold cyan")
        header_text.append(f"  |  {now}", style="dim")
        return Panel(header_text, style="bold blue")
    
    def _create_prices_table(self) -> Table:
        """Create market prices table."""
        table = Table(title="[PRICES]", box=box.ROUNDED, show_header=True)
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Price", justify="right", style="white", width=12)
        table.add_column("Trend", justify="center", width=12)
        table.add_column("RSI", justify="right", width=8)
        table.add_column("MACD", justify="center", width=10)
        table.add_column("BB %", justify="right", width=8)
        
        for symbol in config.TRADING_SYMBOLS:
            ind = self._indicators.get(symbol)
            if not ind:
                table.add_row(symbol, "---", "---", "---", "---", "---")
                continue
            
            # RSI color
            rsi_style = "green" if ind.rsi < 30 else "red" if ind.rsi > 70 else "white"
            
            # MACD signal
            macd_signal = "[+] BUY" if ind.macd_histogram > 0 else "[-] SELL"
            
            # Trend indicator
            trend_style = "green" if "up" in ind.trend_strength else "red" if "down" in ind.trend_strength else "yellow"
            trend_text = "UP" if "up" in ind.trend_strength else "DN" if "down" in ind.trend_strength else "--"
            
            table.add_row(
                symbol,
                f"${ind.price:,.2f}",
                Text(f"[{trend_text}] {ind.trend_strength.upper()}", style=trend_style),
                Text(f"{ind.rsi:.1f}", style=rsi_style),
                macd_signal,
                f"{ind.bb_percent:.2f}"
            )
        
        return table
    
    def _create_signals_table(self) -> Table:
        """Create trading signals table with SMC institutional signals."""
        table = Table(title="[SIGNALS]", box=box.ROUNDED, show_header=True)
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Signal", justify="center", width=14)
        table.add_column("Conf", justify="right", width=5)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("SL", justify="right", width=10)
        table.add_column("TP", justify="right", width=10)
        table.add_column("R:R", justify="right", width=5)
        table.add_column("Status", justify="left", width=12)
        
        for symbol, signal in self._signals.items():
            color = self._get_signal_color(signal)
            
            # Show signal type with strength indicator
            if signal.signal_type == SignalType.NEUTRAL:
                signal_text = "WAITING"
                color = "dim"
                # Show blocking reason if available
                status = signal.reasons[0][:12] if signal.reasons else "Analyzing..."
            else:
                strength_mark = "🔥" if signal.strength == SignalStrength.STRONG else "⚡" if signal.strength == SignalStrength.MODERATE else "•"
                signal_text = f"{strength_mark} {signal.signal_type.value}"
                # Check if SMC confirmed
                smc_confirmed = any("SMC" in r for r in signal.reasons)
                status = "🏦 SMC" if smc_confirmed else "✓ Ready"
            
            # Format prices (0 means not set)
            sl_text = f"${signal.stop_loss:,.0f}" if signal.stop_loss > 0 else "---"
            tp_text = f"${signal.take_profit:,.0f}" if signal.take_profit > 0 else "---"
            rr_text = f"{signal.risk_reward_ratio:.1f}" if signal.stop_loss > 0 else "---"
            
            table.add_row(
                symbol,
                Text(signal_text, style=f"bold {color}"),
                Text(f"{signal.confidence}%", style=color),
                f"${signal.entry_price:,.0f}",
                sl_text,
                tp_text,
                rr_text,
                Text(status, style="cyan" if "SMC" in status else "dim")
            )
        
        if not self._signals:
            table.add_row("---", "No data", "---", "---", "---", "---", "---", "Loading...")
        
        return table
    
    def _create_positions_table(self) -> Table:
        """Create open positions table."""
        table = Table(title="[POSITIONS]", box=box.ROUNDED, show_header=True)
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Side", justify="center", width=8)
        table.add_column("Entry", justify="right", width=12)
        table.add_column("Current", justify="right", width=12)
        table.add_column("P&L %", justify="right", width=10)
        table.add_column("SL", justify="right", width=10)
        table.add_column("TP", justify="right", width=10)
        table.add_column("Duration", justify="right", width=8)
        
        for trade in self._trades:
            # Support both old TradePosition (status) and new Trade (state)
            if hasattr(trade, 'status'):
                if trade.status != TradeStatus.OPEN:
                    continue
            elif hasattr(trade, 'state'):
                # New Trade object uses state
                from strategy.advanced_trade_manager import TradeState
                if trade.state != TradeState.ACTIVE:
                    continue
            
            # Support both side (old) and direction (new)
            side = getattr(trade, 'side', None) or (trade.direction.value if hasattr(trade, 'direction') else 'LONG')
            side_color = "green" if 'LONG' in str(side).upper() else "red"
            pnl_color = self._get_pnl_color(trade.pnl_percent)
            current_price = self._prices.get(trade.symbol, trade.entry_price)
            
            table.add_row(
                trade.symbol,
                Text(str(side).upper(), style=f"bold {side_color}"),
                f"${trade.entry_price:,.2f}",
                f"${current_price:,.2f}",
                Text(f"{trade.pnl_percent:+.2f}%", style=f"bold {pnl_color}"),
                f"${trade.stop_loss:,.2f}" if trade.stop_loss is not None else "---",
                f"${trade.take_profit:,.2f}" if trade.take_profit is not None else "---",
                f"{trade.duration_minutes}m"
            )
        
        # Check for open trades with both status and state
        has_open = False
        for t in self._trades:
            if hasattr(t, 'status') and t.status == TradeStatus.OPEN:
                has_open = True
                break
            elif hasattr(t, 'state'):
                from strategy.advanced_trade_manager import TradeState
                if t.state == TradeState.ACTIVE:
                    has_open = True
                    break
        
        if not self._trades or not has_open:
            table.add_row("---", "No open positions", "---", "---", "---", "---", "---", "---")
        
        return table
    
    def _create_stats_panel(self) -> Panel:
        """Create trading statistics panel with wallet info."""
        stats = self._stats or {}
        
        content = Text()
        
        # === WALLET INFO (PROMINENT) ===
        balance = stats.get('account_balance', 0)
        content.append("💰 Wallet: ", style="bold white")
        balance_color = "bold green" if balance >= 100 else "bold yellow" if balance >= 50 else "bold red"
        content.append(f"${balance:,.2f}\n", style=balance_color)
        
        # Available margin and margin used
        available = stats.get('available_balance', balance)
        margin_used = stats.get('margin_used', balance - available)
        if margin_used > 0:
            content.append("📊 In Use: ", style="dim")
            content.append(f"${margin_used:,.2f}\n", style="yellow")
        
        # Drawdown indicator
        drawdown = stats.get('current_drawdown', 0)
        if drawdown > 0:
            dd_color = "red" if drawdown > 20 else "yellow" if drawdown > 10 else "dim"
            content.append("📉 Drawdown: ", style="dim")
            content.append(f"{drawdown:.1f}%\n", style=dd_color)
        
        content.append("─" * 18 + "\n", style="dim")
        
        # === TRADING STATS ===
        content.append("Daily Trades: ", style="dim")
        content.append(f"{stats.get('daily_trades', 0)}/{config.MAX_DAILY_TRADES}\n", style="cyan")
        
        daily_pnl = stats.get('daily_pnl', 0)
        pnl_color = self._get_pnl_color(daily_pnl)
        content.append("Daily P&L: ", style="dim")
        content.append(f"${daily_pnl:+,.2f}\n", style=pnl_color)
        
        content.append("Win Rate: ", style="dim")
        content.append(f"{stats.get('win_rate', 0):.1f}%\n", style="cyan")
        
        content.append("Total Trades: ", style="dim")
        content.append(f"{stats.get('total_trades', 0)}\n", style="cyan")
        
        # Risk info
        content.append("─" * 18 + "\n", style="dim")
        risk_pct = config.RISK_PER_TRADE if hasattr(config, 'RISK_PER_TRADE') else 0.02
        content.append("Risk/Trade: ", style="dim")
        content.append(f"{risk_pct*100:.0f}% (${balance*risk_pct:.2f})\n", style="cyan")
        
        return Panel(content, title="[STATS]", box=box.ROUNDED)
    
    def _create_signal_details_panel(self) -> Panel:
        """Create panel with signal reasoning including SMC analysis."""
        content = Text()
        
        for symbol, signal in self._signals.items():
            color = self._get_signal_color(signal)
            
            # Show all signals, not just active ones
            if signal.signal_type == SignalType.NEUTRAL:
                content.append(f"\n{symbol} - ", style="dim")
                content.append("WAITING\n", style="yellow")
            else:
                content.append(f"\n{symbol} - ", style=f"bold {color}")
                content.append(f"{signal.signal_type.value}\n", style=f"bold {color}")
            
            # Show reasons with SMC highlighted
            for reason in signal.reasons[:5]:  # Show up to 5 reasons
                if "SMC" in reason or "Liquidity" in reason:
                    content.append(f"  🏦 {reason}\n", style="cyan bold")
                elif "BLOCKED" in reason or "CONFLICT" in reason:
                    content.append(f"  ⚠ {reason}\n", style="red")
                else:
                    content.append(f"  • {reason}\n", style="dim")
        
        if not content.plain:
            content.append("Analyzing market...", style="dim")
        
        return Panel(content, title="[ANALYSIS]", box=box.ROUNDED)
    
    def _create_messages_panel(self) -> Panel:
        """Create system messages panel."""
        content = Text()
        
        for msg in self._messages[-5:]:  # Last 5 messages
            content.append(f"• {msg}\n", style="dim")
        
        if not self._messages:
            content.append("System ready...", style="dim")
        
        return Panel(content, title="[MESSAGES]", box=box.ROUNDED)
    
    def update(self, 
               prices: Dict[str, float] = None,
               indicators: Dict[str, IndicatorValues] = None,
               signals: Dict[str, TradeSignal] = None,
               trades: List[Trade] = None,
               stats: Dict = None,
               message: str = None):
        """Update dashboard data."""
        if prices:
            self._prices.update(prices)
        if indicators:
            self._indicators.update(indicators)
        if signals:
            self._signals.update(signals)
        if trades is not None:
            self._trades = trades
        if stats:
            self._stats = stats
        if message:
            self._messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
            if len(self._messages) > 50:
                self._messages = self._messages[-50:]
        
        self.last_update = datetime.now()
    
    def _create_footer_panel(self) -> Panel:
        """Create footer panel with system status."""
        now = datetime.now().strftime("%H:%M:%S")
        uptime = (datetime.now() - getattr(self, '_start_time', datetime.now())).seconds // 60
        
        content = Text()
        content.append("System Status: ", style="dim")
        content.append("RUNNING", style="bold green")
        content.append(f"  |  Uptime: {uptime}m", style="dim")
        content.append(f"  |  Last Update: {now}", style="dim")
        content.append(f"  |  Symbols: {', '.join(config.TRADING_SYMBOLS)}", style="dim")
        content.append("\n")
        content.append("Press ", style="dim")
        content.append("Ctrl+C", style="bold yellow")
        content.append(" to stop", style="dim")
        
        return Panel(content, title="[SYSTEM]", box=box.ROUNDED, style="dim")
    
    def render(self) -> Layout:
        """Render the complete dashboard layout."""
        layout = Layout()
        
        # Main structure - removed footer, integrated messages into main
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        # Main area split
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Left column - market data
        layout["left"].split_column(
            Layout(name="prices"),
            Layout(name="signals"),
            Layout(name="positions")
        )
        
        # Right column - info panels
        layout["right"].split_column(
            Layout(name="stats"),
            Layout(name="analysis"),
            Layout(name="messages")
        )
        
        # Populate layout
        layout["header"].update(self._create_header())
        layout["prices"].update(self._create_prices_table())
        layout["signals"].update(self._create_signals_table())
        layout["positions"].update(self._create_positions_table())
        layout["stats"].update(self._create_stats_panel())
        layout["analysis"].update(self._create_signal_details_panel())
        layout["messages"].update(self._create_messages_panel())
        layout["footer"].update(self._create_footer_panel())
        
        return layout
    
    def run_live(self, update_callback=None, refresh_rate: float = 1.0):
        """
        Run dashboard with live updates.
        
        Args:
            update_callback: Function to call for data updates
            refresh_rate: Seconds between refreshes
        """
        self.console.clear()
        
        with Live(self.render(), console=self.console, 
                  refresh_per_second=1/refresh_rate, screen=True) as live:
            try:
                while True:
                    if update_callback:
                        update_callback()
                    live.update(self.render())
                    time.sleep(refresh_rate)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Dashboard stopped.[/yellow]")
    
    def print_signal_alert(self, signal: TradeSignal):
        """Print a highlighted signal alert."""
        color = self._get_signal_color(signal)
        
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold {color}]{signal.signal_type.value}[/] {signal.symbol}\n"
            f"Entry: ${signal.entry_price:,.2f} | SL: ${signal.stop_loss:,.2f} | TP: ${signal.take_profit:,.2f}\n"
            f"Confidence: {signal.confidence}% | R:R {signal.risk_reward_ratio:.1f}\n"
            f"Reasons: {', '.join(signal.reasons[:3])}",
            title="!! TRADE SIGNAL !!",
            border_style=color
        ))
    
    def print_exit_alert(self, trade: Trade, reason: str):
        """Print exit alert for a position."""
        pnl_color = self._get_pnl_color(trade.pnl_percent)
        
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold yellow]EXIT[/] {trade.symbol} ({trade.side.upper()})\n"
            f"Entry: ${trade.entry_price:,.2f} → Exit: ${trade.exit_price or 0:,.2f}\n"
            f"[{pnl_color}]P&L: {trade.pnl_percent:+.2f}%[/]\n"
            f"Reason: {reason}",
            title="!! EXIT SIGNAL !!",
            border_style="yellow"
        ))


# Singleton instance
dashboard = Dashboard()
