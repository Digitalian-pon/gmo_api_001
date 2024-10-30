import datetime
import numpy as np
import talib
from app.models.candle import factory_candle_class
from app.models.dfcandle import DataFrameCandle
from app.models.events import SignalEvents
from gmocoin.gmocoin import APIClient
import constants
import settings
import requests
import json
import hmac
import hashlib
import time
import logging

# ログ設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# 取引時間を秒に変換する関数
def duration_seconds(duration: str) -> int:
    if duration == constants.DURATION_1M:
        return 60
    if duration == constants.DURATION_5M:
        return 60 * 5
    if duration == constants.DURATION_1H:
        return 60 * 60
    else:
        return 0

# シグナル検出関数
def determine_trade_action(df, optimized_trade_params):
    if optimized_trade_params is None:
        logger.warning("Optimized trade parameters are not set. No trade action will be performed.")
        return None

    params = optimized_trade_params
    buy_signal, sell_signal = False, False
    signal_strength = 0

    # EMAシグナル
    if params.ema_enable:
        ema_values_1 = talib.EMA(np.array(df.closes), params.ema_period_1)
        ema_values_2 = talib.EMA(np.array(df.closes), params.ema_period_2)
        if len(ema_values_1) >= 2 and len(ema_values_2) >= 2:
            ema_diff = ema_values_2[-1] - ema_values_1[-1]
            buy_signal = ema_values_1[-2] < ema_values_2[-2] and ema_values_1[-1] >= ema_values_2[-1] and ema_diff > 0.05
            sell_signal = ema_values_1[-2] > ema_values_2[-2] and ema_values_1[-1] <= ema_values_2[-1] and ema_diff > 0.05
            signal_strength += 1 if buy_signal or sell_signal else 0

        logger.debug(f"EMA Period 1: {ema_values_1[-2:]}, EMA Period 2: {ema_values_2[-2:]}")
        logger.debug(f"EMA Buy Signal: {buy_signal}, EMA Sell Signal: {sell_signal}")

    # ボリンジャーバンドのシグナル生成
    if params.bb_enable:
        upper_band, middle_band, lower_band = talib.BBANDS(np.array(df.closes), timeperiod=params.bb_n, nbdevup=params.bb_k, nbdevdn=params.bb_k)
        last_close = df.closes[-1]
        if last_close > upper_band[-1]:
            sell_signal = True
            signal_strength += 1
        elif last_close < lower_band[-1]:
            buy_signal = True
            signal_strength += 1

        logger.debug(f"Bollinger Bands - Last Close: {last_close}, Upper Band: {upper_band[-1]}, Lower Band: {lower_band[-1]}")
        logger.debug(f"Bollinger Bands Buy Signal: {buy_signal}, Sell Signal: {sell_signal}")

    # MACDシグナル
    if params.macd_enable:
        macd, macdsignal, macdhist = talib.MACD(np.array(df.closes), fastperiod=params.macd_fast_period, slowperiod=params.macd_slow_period, signalperiod=params.macd_signal_period)
        if len(macd) >= 2 and len(macdsignal) >= 2:
            buy_signal = macd[-1] > macdsignal[-1] and macd[-2] <= macdsignal[-2]
            sell_signal = macd[-1] < macdsignal[-1] and macd[-2] >= macdsignal[-2]
            signal_strength += 1 if buy_signal or sell_signal else 0

        logger.debug(f"MACD: {macd[-2:]}, MACD Signal: {macdsignal[-2:]}")
        logger.debug(f"MACD Buy Signal: {buy_signal}, MACD Sell Signal: {sell_signal}")

    if signal_strength >= 2:  # シグナルが強い場合のみ取引
        if buy_signal:
            logger.info("Buy signal generated")
            return "BUY"
        elif sell_signal:
            logger.info("Sell signal generated")
            return "SELL"
    else:
        logger.debug("No strong trading signal detected")
        return None

# 取引実行関数
def execute_trade_action(action, candle, ai_instance):
    if action == "BUY":
        logger.info("Executing Buy Order")
        ai_instance.buy(candle)
    elif action == "SELL":
        logger.info("Executing Sell Order")
        ai_instance.sell(candle)

# AIクラスの定義
class AI(object):

    def __init__(self, product_code, use_percent, duration, past_period, stop_limit_percent, back_test):
        self.API = APIClient(settings.api_key, settings.api_secret)
        self.product_code = product_code
        self.use_percent = use_percent
        self.duration = duration
        self.past_period = past_period
        self.optimized_trade_params = None
        self.stop_limit = 0
        self.stop_limit_percent = stop_limit_percent
        self.back_test = back_test
        self.start_trade = datetime.datetime.utcnow()
        self.signal_events = SignalEvents() if back_test else SignalEvents.get_signal_events_by_count(1)
        self.candle_cls = factory_candle_class(self.product_code, self.duration)
        self.update_optimize_params(False)

    def update_optimize_params(self, is_continue: bool):
        logger.info('action=update_optimize_params status=run')
        df = DataFrameCandle(self.product_code, self.duration)
        df.set_all_candles(self.past_period)
        if df.candles:
            self.optimized_trade_params = df.optimize_params()
        if self.optimized_trade_params is not None:
            logger.info(f'action=update_optimize_params params={self.optimized_trade_params.__dict__}')
        else:
            logger.warning("No optimized trade parameters found after update.")

        if is_continue and self.optimized_trade_params is None:
            time.sleep(10 * duration_seconds(self.duration))
            self.update_optimize_params(is_continue)

    def create_headers(self, method, path, body):
        timestamp = str(int(time.time() * 1000))
        text = timestamp + method + path + body
        sign = hmac.new(settings.api_secret.encode(), text.encode(), hashlib.sha256).hexdigest()
        return {
            "API-KEY": settings.api_key,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign,
            "Content-Type": "application/json"
        }

    def buy(self, candle):
        if self.back_test:
            could_buy = self.signal_events.buy(self.product_code, candle.time, candle.close, 1.0, save=False)
            return could_buy

        if self.start_trade > candle.time:
            logger.warning('action=buy status=false error=old_time')
            return False

        balance = self.API.get_balance()
        if not balance:
            logger.error('Failed to retrieve balance.')
            return False

        units = int(balance['available'] * self.use_percent)
        req_body = json.dumps({
            "symbol": self.product_code,
            "side": "BUY",
            "executionType": "LIMIT",
            "size": str(units),
            "price": candle.close
        })

        path = "/v1/order"
        headers = self.create_headers("POST", path, req_body)
        res = requests.post(settings.api_url + path, headers=headers, data=req_body)
        trade = res.json()

        logger.info(f'Buy order response: {trade}')
        could_buy = self.signal_events.buy(self.product_code, candle.time, candle.close, units, save=True)
        return could_buy

    def sell(self, candle):
        if self.back_test:
            could_sell = self.signal_events.sell(self.product_code, candle.time, candle.close, 1.0, save=False)
            return could_sell

        if self.start_trade > candle.time:
            logger.warning('action=sell status=false error=old_time')
            return False

        trades = self.API.get_open_trade()
        if not trades:
            logger.error('Failed to fetch open trades.')
            return False

        sum_price = 0
        units = 0
        for trade in trades:
            closed_trade = self.API.trade_close(trade['trade_id'])
            sum_price += closed_trade['price'] * abs(closed_trade['units'])
            units += abs(closed_trade['units'])

        could_sell = self.signal_events.sell(self.product_code, candle.time, sum_price / units, units, save=True)
        return could_sell

    def trade(self):
        logger.info('action=trade status=run')
        if self.optimized_trade_params is None:
            self.update_optimize_params(False)
            if self.optimized_trade_params is None:
                logger.warning("Optimized trade parameters are still not set after update.")
                return

        df = DataFrameCandle(self.product_code, self.duration)
        df.set_all_candles(self.past_period)

        action = determine_trade_action(df, self.optimized_trade_params)
        if action:
            execute_trade_action(action, df.candles[-1], self)
