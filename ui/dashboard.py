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
        """Create trading signals table."""
        table = Table(title="[SIGNALS]", box=box.ROUNDED, show_header=True)
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Signal", justify="center", width=12)
        table.add_column("Confidence", justify="right", width=10)
        table.add_column("Entry", justify="right", width=12)
        table.add_column("Stop Loss", justify="right", width=12)
        table.add_column("Take Profit", justify="right", width=12)
        table.add_column("R:R", justify="right", width=6)
        
        for symbol, signal in self._signals.items():
            if signal.signal_type == SignalType.NEUTRAL:
                continue
            
            color = self._get_signal_color(signal)
            strength_mark = "***" if signal.strength == SignalStrength.STRONG else "**" if signal.strength == SignalStrength.MODERATE else "*"
            
            table.add_row(
                symbol,
                Text(f"{strength_mark} {signal.signal_type.value}", style=f"bold {color}"),
                Text(f"{signal.confidence}%", style=color),
                f"${signal.entry_price:,.2f}",
                f"${signal.stop_loss:,.2f}",
                f"${signal.take_profit:,.2f}",
                f"{signal.risk_reward_ratio:.1f}"
            )
        
        if not any(s.signal_type != SignalType.NEUTRAL for s in self._signals.values()):
            table.add_row("---", "No signals", "---", "---", "---", "---", "---")
        
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
        """Create trading statistics panel."""
        stats = self._stats or {}
        
        content = Text()
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
        
        return Panel(content, title="[STATS]", box=box.ROUNDED)
    
    def _create_signal_details_panel(self) -> Panel:
        """Create panel with signal reasoning."""
        content = Text()
        
        for symbol, signal in self._signals.items():
            if signal.signal_type == SignalType.NEUTRAL:
                continue
            
            color = self._get_signal_color(signal)
            content.append(f"\n{symbol} - {signal.signal_type.value}\n", style=f"bold {color}")
            
            for reason in signal.reasons[:4]:  # Limit to 4 reasons
                content.append(f"  • {reason}\n", style="dim")
        
        if not content.plain:
            content.append("Waiting for signals...", style="dim")
        
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
