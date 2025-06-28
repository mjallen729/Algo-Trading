signal.price,
                order_type='limit'
            )
            
            if order and order.get('status') == 'filled':
                # Update portfolio
                self.portfolio.add_position(
                    asset=asset,
                    quantity=quantity,
                    entry_price=signal.price,
                    side='long'
                )
                
                # Record trade
                trade_record = {
                    'timestamp': datetime.now(),
                    'asset': asset,
                    'side': 'buy',
                    'quantity': quantity,
                    'price': signal.price,
                    'value': position_size,
                    'signal_confidence': signal.confidence,
                    'signal_metadata': signal.metadata
                }
                self.trade_history.append(trade_record)
                
                logger.info(f"BUY order executed: {asset} @ {signal.price:.2f} (qty: {quantity:.6f})")
            else:
                logger.warning(f"BUY order failed for {asset}")
                
        except Exception as e:
            logger.error(f"Error executing buy order for {asset}: {e}")
    
    async def _execute_sell_order(self, asset: str, position_size: float, signal: TradingSignal):
        """Execute a sell order."""
        try:
            # Check if we have a position to sell
            current_position = self.portfolio.get_position(asset)
            
            if current_position and current_position['quantity'] > 0:
                # Sell existing position
                quantity = current_position['quantity']
                
                # Place sell order
                order = await self.broker.place_order(
                    symbol=asset,
                    side='sell',
                    quantity=quantity,
                    price=signal.price,
                    order_type='limit'
                )
                
                if order and order.get('status') == 'filled':
                    # Update portfolio
                    self.portfolio.close_position(asset)
                    
                    # Record trade
                    trade_record = {
                        'timestamp': datetime.now(),
                        'asset': asset,
                        'side': 'sell',
                        'quantity': quantity,
                        'price': signal.price,
                        'value': quantity * signal.price,
                        'signal_confidence': signal.confidence,
                        'signal_metadata': signal.metadata
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"SELL order executed: {asset} @ {signal.price:.2f} (qty: {quantity:.6f})")
                else:
                    logger.warning(f"SELL order failed for {asset}")
            else:
                logger.info(f"No position to sell for {asset}")
                
        except Exception as e:
            logger.error(f"Error executing sell order for {asset}: {e}")
    
    async def close_all_positions(self):
        """Close all open positions."""
        logger.info("Closing all positions...")
        
        try:
            positions = self.portfolio.get_all_positions()
            
            for asset, position in positions.items():
                if position['quantity'] > 0:
                    # Get current market price (simplified - using last known price)
                    current_price = position['current_price']
                    
                    # Place sell order
                    order = await self.broker.place_order(
                        symbol=asset,
                        side='sell',
                        quantity=position['quantity'],
                        price=current_price,
                        order_type='market'
                    )
                    
                    if order and order.get('status') == 'filled':
                        self.portfolio.close_position(asset)
                        logger.info(f"Closed position: {asset}")
            
            logger.info("All positions closed")
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    async def shutdown(self):
        """Shutdown the trading engine."""
        logger.info("Shutting down trading engine...")
        
        try:
            # Close all positions
            await self.close_all_positions()
            
            # Shutdown broker connection
            await self.broker.shutdown()
            
            logger.info("Trading engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during trading engine shutdown: {e}")
    
    def get_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            portfolio_metrics = self.portfolio.get_performance_metrics()
            
            # Trade statistics
            total_trades = len(self.trade_history)
            buy_trades = len([t for t in self.trade_history if t['side'] == 'buy'])
            sell_trades = len([t for t in self.trade_history if t['side'] == 'sell'])
            
            # Calculate win rate (simplified)
            winning_trades = 0
            if sell_trades > 0:
                for trade in self.trade_history:
                    if trade['side'] == 'sell':
                        # Find corresponding buy trade
                        buy_trades_for_asset = [
                            t for t in self.trade_history 
                            if t['asset'] == trade['asset'] and t['side'] == 'buy' 
                            and t['timestamp'] < trade['timestamp']
                        ]
                        
                        if buy_trades_for_asset:
                            latest_buy = max(buy_trades_for_asset, key=lambda x: x['timestamp'])
                            if trade['price'] > latest_buy['price']:
                                winning_trades += 1
            
            win_rate = winning_trades / sell_trades if sell_trades > 0 else 0
            
            # Average signal confidence
            avg_confidence = sum(t['signal_confidence'] for t in self.trade_history) / total_trades if total_trades > 0 else 0
            
            report = {
                'portfolio_metrics': portfolio_metrics,
                'trading_stats': {
                    'total_trades': total_trades,
                    'buy_trades': buy_trades,
                    'sell_trades': sell_trades,
                    'win_rate': win_rate,
                    'avg_signal_confidence': avg_confidence
                },
                'execution_mode': self.execution_mode,
                'report_timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def get_status(self) -> Dict:
        """Get current trading engine status."""
        return {
            'is_initialized': self.is_initialized,
            'execution_mode': self.execution_mode,
            'total_trades': len(self.trade_history),
            'portfolio_value': self.portfolio.get_total_value(),
            'available_capital': self.portfolio.get_available_capital(),
            'active_positions': len(self.portfolio.get_all_positions())
        }
    
    def get_trade_history(self) -> List[Dict]:
        """Get complete trade history."""
        return self.trade_history.copy()
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades."""
        return self.trade_history[-limit:] if self.trade_history else []
