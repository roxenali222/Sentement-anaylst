import pandas as pd
from datetime import timedelta


signals = pd.read_csv('stochastic_4h_signals.csv', parse_dates=['timestamp'])[['timestamp','open', 'high', 'low', 'close', 'SC_Signals','Signal','%K','%D']]
df = pd.read_csv('ohlc_1m_data_last_day.csv', parse_dates=['timestamp'])


import time
from turtle import back
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
def ledger(signals, df, time_frame='4H'):
    """
    This function takes a dataframe of signals which is 4h interval data and df which is 1min interval data.
    """
    signals = signals
    df_1min = df
    action = []
    signal = []
    date_time = []
    buy_price = []
    sell_price = []
    pnl = []
    transaction_fee = 0.07 / 100
    profit_percentage = 0.05
    stop_loss_percentage = 0.02
    next_trade_time = None  # To store the time of the next trade

    def calculate_levels(buy, direction):
        if direction == 1:
            tp_price = buy + (buy * profit_percentage)
            sl_price = buy - (buy * stop_loss_percentage)
        elif direction == -1:
            tp_price = buy - (buy * profit_percentage)
            sl_price = buy + (buy * stop_loss_percentage)
        else:
            tp_price = sl_price = np.nan
        return tp_price, sl_price

    for i in range(len(signals)):
        if len(action) == 0 or (next_trade_time is not None and signals['timestamp'][i] >= next_trade_time):
            # Check and open a trade in the signal's direction
            if signals['SC_Signals'][i] != 0:
                action.append(signals['SC_Signals'][i])
                date_time.append(signals['timestamp'][i])
                buy_price.append(signals['open'][i])
                buy = signals['open'][i]
                if signals['SC_Signals'][i] == 1:
                    signal.append('Buy')
                elif signals['SC_Signals'][i] == -1:
                    signal.append('Sell')
                sell_price.append(np.nan)
                pnl.append(np.nan)
                next_trade_time = None
                tp_price, sl_price = calculate_levels(buy, signals['SC_Signals'][i])
        else:
            if action[-1] == signals.iloc[i].Signal or signals['SC_Signals'][i] == 0:
                if action[-1] == 1:
                    tp_price, sl_price = calculate_levels(buy, 1)
                elif action[-1] == -1:
                    tp_price, sl_price = calculate_levels(buy, -1)
                else:
                    continue  
                if i + 1 < len(signals):
                    next_signal_date = signals.loc[i + 1, 'timestamp']
                else:
                    next_signal_date = df_1min['timestamp'].max()

                end_datetime = signals.iloc[i]['timestamp']
                current_datetime = end_datetime - timedelta(hours=int(time_frame[0]))
                df_1min_slice = df_1min[(df_1min['timestamp'] >= current_datetime) & (df_1min['timestamp'] <= end_datetime)]
                df_1min_slice.reset_index(drop=True, inplace=True)

                for j in df_1min_slice.iterrows():
                    if j[1].high > tp_price and action[-1] == 1:
                        # Take profit
                        time = j[1].timestamp
                        sell_price.append(j[1].high)
                        action.append(action[-1])
                        signal.append('TP Hit')
                        date_time.append(time)
                        buy_price.append(np.nan)
                        pnl.append(np.nan)
                        next_trade_time = (pd.to_datetime(time) + pd.Timedelta(hours=(int(time_frame[0]) - pd.to_datetime(time).hour % 4))).replace(minute=0, second=0, microsecond=0)
                        break
                    elif j[1].low < sl_price and action[-1] == 1:
                        # Stop loss
                        time = j[1].timestamp
                        sell_price.append(j[1].low)
                        signal.append('SL Hit')
                        action.append(action[-1])
                        date_time.append(time)
                        buy_price.append(np.nan)
                        pnl.append(np.nan)
                        next_trade_time = (pd.to_datetime(time) + pd.Timedelta(hours=(int(time_frame[0]) - pd.to_datetime(time).hour % 4))).replace(minute=0, second=0, microsecond=0)
                        break
                    elif j[1].low < tp_price and action[-1] == -1:
                        # Take profit
                        time = j[1].timestamp
                        sell_price.append(j[1].low)
                        action.append(action[-1])
                        signal.append('TP Hit')
                        date_time.append(time)
                        buy_price.append(np.nan)
                        pnl.append(np.nan)
                        next_trade_time = (pd.to_datetime(time) + pd.Timedelta(hours=(int(time_frame[0]) - pd.to_datetime(time).hour % 4))).replace(minute=0, second=0, microsecond=0)
                        break
                    elif j[1].high > sl_price and action[-1] == -1:
                        # Stop loss
                        time = j[1].timestamp
                        sell_price.append(j[1].high)
                        action.append(action[-1])
                        signal.append('SL Hit')
                        date_time.append(time)
                        buy_price.append(np.nan)
                        pnl.append(np.nan)
                        next_trade_time = (pd.to_datetime(time) + pd.Timedelta(hours=(int(time_frame[0]) - pd.to_datetime(time).hour % 4))).replace(minute=0, second=0, microsecond=0)
                        break
            else:
                # Close the previous trade due to trend change
                if action[-1] in ['tp hit', 'sl hit']:
                    if signals['SC_Signals'][i] == 0:
                        continue  # Ignore if the new signal is 0
                    action.append(signals['SC_Signals'][i])
                    date_time.append(signals['timestamp'][i])
                    buy_price.append(signals['open'][i])
                    signal.append('Buy')
                    buy = signals['open'][i]
                    sell_price.append(np.nan)
                    pnl.append(np.nan)
                    next_trade_time = None
                else:
                    action.append(signals['SC_Signals'][i])
                    date_time.append(signals['timestamp'][i])
                    sell_price.append(signals['open'][i])
                    buy = signals['open'][i]
                    if signals['SC_Signals'][i] == 1:
                        signal.append('Buy')
                    elif signals['SC_Signals'][i] == -1:
                        signal.append('Sell')
                    buy_price.append(np.nan)
                    pnl.append(np.nan)

                    action.append(signals['SC_Signals'][i])
                    date_time.append(signals['timestamp'][i])
                    buy_price.append(signals['open'][i])
                    buy = signals['open'][i]
                    if signals['SC_Signals'][i] == 1:
                        signal.append('Buy')
                    elif signals['SC_Signals'][i] == -1:
                        signal.append('Sell')
                    sell_price.append(np.nan)
                    pnl.append(np.nan)
                    
    # Creating the resulting dataframe
    ledger_df = pd.DataFrame({
        'date_time': date_time,
        'action': action,
        'signal': signal,
        'buy_price': buy_price,
        'sell_price': sell_price,
        'pnl': pnl
    })
    
    print("Ledger DataFrame columns:", ledger_df.columns)  # Debugging line
    print("Ledger DataFrame head:\n", ledger_df.head())  # Debugging line
    
    ledger_df['buy_price'] = ledger_df['buy_price'].fillna(method='ffill')
    
    def calculate_pnl(df):
        balance = 1000
        df['balance'] = np.nan
        df['pnl'] = np.nan
        for index, row in df.iterrows():
            buy_price = float(row['buy_price']) if not pd.isna(row['buy_price']) else 0
            sell_price = float(row['sell_price']) if not pd.isna(row['sell_price']) else 0
            pnl_percent = 0  # Initialize pnl_percent

            if pd.isna(sell_price) or pd.isna(buy_price):
                df.at[index, 'balance'] = balance
            else:
                if row['action'] == 1:  # Long
                    pnl_percent = ((sell_price - buy_price) / buy_price) * 100
                    balance += balance * (pnl_percent / 100) - transaction_fee
                elif row['action'] == -1:  # Short
                    pnl_percent = ((buy_price - sell_price) / buy_price) * 100
                    balance += balance * (pnl_percent / 100) - transaction_fee
                
                df.at[index, 'balance'] = balance
                df.at[index, 'pnl'] = round(pnl_percent, 2)  # Efficient assignment

        return df

    ledger_df = calculate_pnl(ledger_df)
    
    return ledger_df


ledger_df = ledger(signals, df)