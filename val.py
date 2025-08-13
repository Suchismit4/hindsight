from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import backtrader as bt
import pandas as pd

import logging
logging.basicConfig(
    filename='backtrader_strategy.log',  
    filemode='w',                        
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


data = pd.read_csv('BTCUSDT.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
print("From raw data: ", data.loc['2020-01-07 13:00:00'])
data.dropna(inplace=True)
data.to_csv('BTCUSDT_backtrader.csv')

class BarProfitFactor(bt.Analyzer):
    """
    Profit Factor computed from bar-by-bar changes in total portfolio value (equity),
    as described by Timothy Masters: PF = sum(positive Δequity) / sum(|negative Δequity|).
    Includes commissions/slippage as reflected in broker.getvalue().
    """

    params = dict(
        ignore_zeros=False
    )

    def start(self):
        self._last_value = None
        self._gains = 0.0
        self._losses = 0.0

    def next(self):
        v = float(self.strategy.broker.getvalue())  # equity at *close* of current bar
        if self._last_value is None:
            self._last_value = v
            return

        delta = v - self._last_value

        if delta > 0:
            self._gains += delta
        elif delta < 0:
            self._losses += -delta
        else:
            if not self.p.ignore_zeros:
                # treat zero as neither gain nor loss (Masters typically ignores zeros)
                pass

        self._last_value = v

    def get_analysis(self):
        if self._gains == 0 and self._losses == 0:
            pf = 1.0  # flat equity curve
        elif self._losses == 0:
            pf = math.inf
        else:
            pf = self._gains / self._losses

        return dict(
            gains=self._gains,
            losses=self._losses,
            profitfactor=pf
        )

#  strategy definition 
class TestStrategy(bt.Strategy):
    params = dict(
        sma1_period=100,
        sma2_period=200,
        target_units=1,   
    )

    def log(self, txt, dt=None):
        """Logging method writes to both console and file."""
        dt = dt or self.datas[0].datetime.datetime(0)
        msg = f'{dt.isoformat()}, {txt}'
        # writes to your file (and to console if you want)
        self.logger.info(msg)

    def __init__(self):
        self.dataclose = self.datas[0].close

        # two simple moving averages
        self.sma100 = bt.ind.SMA(self.dataclose, period=self.p.sma1_period)
        self.sma200 = bt.ind.SMA(self.dataclose, period=self.p.sma2_period)

        self.logger = logging.getLogger(self.__class__.__name__)

        # keep track of pending orders 
        self.order = None

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        close = self.dataclose[0]
        _open = self.datas[0].open[0]

        # yesterday’s vs today’s SMA values
        sma100_y, sma200_y = float(self.sma100[-1]), float(self.sma200[-1])
        sma100_c, sma200_c = float(self.sma100[0]),  float(self.sma200[0])

        # log the close + SMA readings
        # self.log(f'Close: {close:.2f}| OPEN={_open:.2f} | SMA100_prev={sma100_y:.2f}, SMA200_prev={sma200_y:.2f}'
        #          f' | SMA100={sma100_c:.2f}, SMA200={sma200_c:.2f}')

        # current position size (in BTC)
        current_pos = self.position.size

        # your exact crossover logic
        if sma100_y > sma200_y and sma100_c <= sma200_c:
            target = self.p.target_units      # go long 0.5
        elif sma100_y < sma200_y and sma100_c >= sma200_c:
            target = -self.p.target_units                     # flat
        else:
            return                           # no signal

        delta = target - current_pos
        if abs(delta) < 1e-8:
            return  # already at target

        # Place market order to move toward target
        if delta > 0:
            self.log(f'SET BUY ORDER size={delta:.4f} | '
                     f'SMAs@prev:100={sma100_y:.2f},200={sma200_y:.2f} SMA100={sma100_c:.2f},SMA200={sma200_c:.2f}')
            self.order = self.buy(size=delta)
        else:
            self.log(f'SET SELL ORDER size={abs(delta):.4f} | '
                     f'SMAs@prev:100={sma100_y:.2f},200={sma200_y:.2f} SMA100={sma100_c:.2f},SMA200={sma200_c:.2f}')
            self.order = self.sell(size=abs(delta))

    def notify_order(self, order):
        # called on order status changes
        if order.status in [order.Submitted, order.Accepted]:
            return  # still working

        dt = self.datas[0].datetime.datetime(0)
        if order.status == order.Completed:
            side = 'BUY' if order.isbuy() else 'SELL'
            self.log(
                f'{side} EXECUTED size={order.executed.size:.4f} '
                f'at price={order.executed.price:.2f}'
            )

            # Log portfolio value immediately after the trade
            cash = self.broker.getcash()        
            pv = self.broker.getvalue()
            self.log(f'Portfolio Value after trade: {pv:.2f} | Cash: {cash:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # reset
        self.order = None


# cerebro setup & run
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)

    df = pd.read_csv('BTCUSDT_backtrader.csv', parse_dates=['datetime'], index_col='datetime')
    print("From backtrader version: ", df.loc['2020-01-07 12:00:00'])
    df = df.asfreq('h').dropna()   # ensure hourly frequency, no gaps

    # Compute SMA100 and SMA200 on close and add as new columns
    df['sma100'] = df['close'].rolling(window=100).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()

    # Export the updated DataFrame to CSV (includes original columns + smas)
    df.to_csv('BTCUSDT_with_smas.csv')

    data = bt.feeds.PandasData(
        dataname=df,
        timeframe=bt.TimeFrame.Minutes,
        compression=60,
        fromdate=datetime.datetime(2019, 12, 31, 1, 0),
        todate=datetime.datetime(2024, 1, 1, 0, 0),
    )
    cerebro.adddata(data)

    cerebro.broker.setcash(100000.0)
   
    # Sharpe (annualized).
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe_a', riskfreerate=0.01)
    # Max drawdown
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    # Trade stats for Profit Factor
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {final_value:.2f}')

    # Sharpe
    sharpe_res = strat.analyzers.sharpe_a.get_analysis()
    sharpe = sharpe_res.get('sharperatio', None)

    # Max Drawdown
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd_pct = None
    max_dd_len = None
    if 'max' in dd:
        max_dd_pct = dd['max'].get('drawdown', None)     # percent
        max_dd_len = dd['max'].get('len', None)          # bars

    # Profit Factor from TradeAnalyzer
    cerebro.addanalyzer(BarProfitFactor, _name='bar_pf')
    results = cerebro.run()
    strat = results[0]

    bar_pf = strat.analyzers.bar_pf.get_analysis()
    print("Bar PF (Masters):", bar_pf['profitfactor'])
    print("Σ gains:", bar_pf['gains'], "Σ losses:", bar_pf['losses'])
    print("\n==== Performance Metrics ====")
    print(f"Sharpe Ratio (annualized): {sharpe if sharpe is not None else 'N/A'}")
    print(f"Max Drawdown (%): {max_dd_pct if max_dd_pct is not None else 'N/A'}")
    print(f"Max Drawdown Length (bars): {max_dd_len if max_dd_len is not None else 'N/A'}")

    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')