#!/usr/bin/env python
# coding: utf-8

# pylint: disable-msg=redefined-outer-name,missing-function-docstring,missing-module-docstring,line-too-long

# In[ ]:


# imports
import os
import sys
from datetime import datetime
from datetime import date as dt
from datetime import time as tm
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from pyfinviz.screener import Screener
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from trending_stocks import get_stocks_with_15_percent_gain


def clean_stock_data(df):
    df.columns = df.columns.droplevel(-1)
    return df


# In[ ]:


# determine whether MA200 is uptrend or not
def trend_value(nums:list):
    summed_nums = sum(nums)
    multiplied_data = 0
    summed_index = 0
    squared_index = 0

    for index, num in enumerate(nums):
        index += 1
        multiplied_data += index * num
        summed_index += index
        squared_index += index**2

    numerator = (len(nums) * multiplied_data) - (summed_nums * summed_index)
    denominator = (len(nums) * squared_index) - summed_index**2
    if denominator != 0:
        return numerator/denominator

    return 0


# determine whether the ticker fulfills trend template
def trend_template(df):
    # calculate moving averages
    df['MA_50'] = round(df['Close'].rolling(window=50).mean(),2)
    df['MA_150'] = round(df['Close'].rolling(window=150).mean(),2)
    df['MA_200'] = round(df['Close'].rolling(window=200).mean(),2)

    if len(df.index) > 5*52:
        df['52_week_low'] = df['Low'].rolling(window = 5*52).min()
        df['52_week_high'] = df['High'].rolling(window = 5*52).max()
    else:
        df['52_week_low'] = df['Low'].rolling(window = len(df.index)).min()
        df['52_week_high'] = df['High'].rolling(window = len(df.index)).max()


    # condition 1&5: Price is above both 150MA and 200MA & above 50MA
    df['condition_1'] = (df['Close'] > df['MA_150']) & (df['Close'] > df['MA_200']) & (df['Close'] > df['MA_50'])

    # condition 2&4: 150MA is above 200MA & 50MA is above both
    df['condition_2'] = (df['MA_150'] > df['MA_200']) & (df['MA_50'] > df['MA_150'])

    # condition 3: 200MA is trending up for at least 1 month
    slope = df['MA_200'].rolling(window = 20).apply(trend_value)
    df['condition_3'] = slope > 0.0

    # condition 6: Price is at least 30% above 52 week low
    df['condition_6'] = df['Low'] > (df['52_week_low']*1.3)

    # condition 7: Price is at least 25% of 52 week high
    df['condition_7'] = df['High'] > (df['52_week_high']*0.75)

    # condition 9 (additional): The relative strength line, which compares a stock's price performance to that of the S&P 500.
    # An upward trending RS line tells you the stock is outperforming the general market.
    df['RS'] = df['Close']/df_spx['Close']
    slope_rs = df['RS'].rolling(window = 20).apply(trend_value)
    df['condition_8'] = slope > 0.0

    df['Pass'] = df[['condition_1','condition_2','condition_3','condition_6','condition_7','condition_8']].all(axis='columns')

    return df

# determine local maxima and minima
def local_high_low(df):
    local_high = argrelextrema(df['High'].to_numpy(),np.greater,order=10)[0]
    local_low = argrelextrema(df['Low'].to_numpy(),np.less,order=10)[0]

    # eliminate for consecutive highs or lows
    # create adjusted local highs and lows
    i = 0
    j = 0
    local_high_low = []
    adjusted_local_high = []
    adjusted_local_low = []

    while i < len(local_high) and j < len(local_low):
        if local_high[i] < local_low[j]:
            while i < len(local_high):
                if local_high[i] < local_low[j]:
                    i+=1
                else:
                    adjusted_local_high.append(local_high[i-1])
                    break
        elif local_high[i] > local_low[j]:
            while j < len(local_low):
                if local_high[i] > local_low[j]:
                    j+=1
                else:
                    adjusted_local_low.append(local_low[j-1])
                    break
        else:
            i+=1
            j+=1

    # add any remaining elements from local_high or local_low
    if i < len(local_high):
        adjusted_local_high.pop(-1)
        while i < len(local_high):
            if local_high[i] > local_low[j-1]:
                i+=1
            else:
                adjusted_local_high.append(local_high[i-1])
                break
        adjusted_local_high.append(local_high[-1])
        adjusted_local_low.append(local_low[j-1])

    if j < len(local_low):
        adjusted_local_low.pop(-1)
        while j < len(local_low):
            if local_high[i-1] > local_low[j]:
                j+=1
            else:
                adjusted_local_low.append(local_low[j-1])
                break
        adjusted_local_low.append(local_low[-1])
        adjusted_local_high.append(local_high[i-1])
    return adjusted_local_high, adjusted_local_low

# measure the depth of contractions
def contractions(df,local_high,local_low):
    local_high = local_high[::-1]
    local_low = local_low[::-1]

    i = 0
    j = 0
    contraction = []

    while i < len(local_low) and j < len(local_high):
        if local_low[i] > local_high[j]:
            contraction.append(round((df['High'].iloc[local_high].iloc[j] - df['Low'].iloc[local_low].iloc[i]) / df['High'].iloc[local_high].iloc[j] * 100,2))
            i+=1
            j+=1
        else:
            j+=1
    return contraction

# measure number of contractions
def num_of_contractions(contraction):
    new_c = 0
    num_of_contraction = 0
    for c in contraction:
        if c > new_c:
            num_of_contraction+=1
            new_c = c
        else:
            break
    return num_of_contraction

# measure depth of maximum and minimum contraction
def max_min_contraction(contraction,num_of_contractions):
    max_contraction = contraction[num_of_contractions-1]
    min_contraction = contraction[0]
    return max_contraction, min_contraction

# measure days of contraction
def weeks_of_contraction(df,local_high,num_of_contractions):
    week_of_contraction = (len(df.index) - local_high[::-1][num_of_contractions-1]) / 5
    return week_of_contraction

# determine whether the ticker has VCP
def vcp(df):
    # prepare data for contractions measurement
    [local_high, local_low] = local_high_low(df)

    contraction = contractions(df,local_high,local_low)

    # calculate no. of contractions
    num_of_contraction = num_of_contractions(contraction)
    if 2 <= num_of_contraction <= 4:
        flag_num = 1
    else:
        flag_num = 0

    # calculate depth of contractions
    [max_c, min_c] = max_min_contraction(contraction,num_of_contraction)
    if max_c > 50:
        flag_max = 0
    else:
        flag_max = 1
    if min_c <= 15:
        flag_min = 1
    else:
        flag_min = 0

    # calculate weeks of contractions
    week_of_contraction = weeks_of_contraction(df,local_high,num_of_contraction)
    if week_of_contraction >= 2:
        flag_week = 1
    else:
        flag_week = 0

    df['30_day_avg_volume'] = round(df['Volume'].rolling(window = 30).mean(),2)
    df['5_day_avg_volume'] = round(df['Volume'].rolling(window = 5).mean(),2)
    # criteria_2: Volume contraction
    df['vol_contraction'] = df['5_day_avg_volume'] < df['30_day_avg_volume']
    if df['vol_contraction'].iloc[-1] == 1:
        flag_vol = 1
    else:
        flag_vol = 0

    # criteria 3: Not break out yet
    if df['High'].iloc[-1] < df['High'].iloc[local_high].iloc[-1]:
        flag_consolidation = 1
    else:
        flag_consolidation = 0

    if flag_num == 1 & flag_max == 1 & flag_min == 1 & flag_week == 1 & flag_vol == 1 & flag_consolidation == 1:
        flag_final = 1
    else:
        flag_final = 0

    return num_of_contraction,max_c,min_c,week_of_contraction,flag_final

# calculate RS rating
def rs_rating(ticker,rs_list):
    try:
        ticker_index = rs_list.index(ticker)
        rs = round(ticker_index / len(rs_list) * 100,0)
    except ValueError:
        rs = -1
        raise
    return rs


def check_market_closed():
    """"""

    currtime = tm(datetime.now().hour, datetime.now().minute)
    opentime = tm(14,30)
    closetime = tm(21,0)
    # if still trading still print a warning and ask to continue
    if currtime > opentime or currtime < closetime:
        result = input("Still trading, results will be inconsistent, Continue ? [y/N] ")
        if result.lower() != "y":
            sys.exit()


def get_screened_tickers():
    """# Screen stocks from Finviz.com with filters"""

    filters = [Screener.MarketCapOption.SMALL_OVER_USD300MLN,
               Screener.AverageVolumeOption.OVER_100K,
               Screener.PriceOption.OVER_USD2,
               Screener._200DaySimpleMovingAverageOption.SMA200_ABOVE_SMA50,
               Screener._200DaySimpleMovingAverageOption.PRICE_ABOVE_SMA200,
               Screener.IndexOption.S_AND_P_500]

    #pages = list(range(1, 16))

    #stock_list = Screener(filter_options=filters, view_option=Screener.ViewOption.PERFORMANCE, pages=pages )

    #print(f"{stock_list.data_frames.get(1).columns}")
    #print(f"{stock_list.data_frames.get(6).index}")

    ticker_list = []
    for p in range(1,20):
        try:
            stock_list = Screener(filter_options=filters,
                                  view_option=Screener.ViewOption.PERFORMANCE, pages=[p] )
            ticker_table = stock_list.data_frames.get(p)
            ticker_list += ticker_table['Ticker'].to_list()
        except ValueError:
            break

    print(f"{len(ticker_list)} Stock Tickers")

    with open("ticker_list.py", 'w') as tl:
        tl.write(f"ticker_list = {ticker_list}\n")

    return ticker_list


def get_saved_ticker_list():
    """"""
    try:
        from ticker_list import ticker_list
        print(f"{len(ticker_list)} Stock Tickers imported")
        return ticker_list
    except ImportError:
        return []


#pages = list(range(1, 26))

# for condition_8, RS rating should be greater than 70
# The RS Rating tracks a stock's share price performance over the last 52 weeks,
# and then compares the result to that of all other stocks.
#performance_table = Screener(filter_options=[Screener.IndexOption.S_AND_P_500], view_option=Screener.ViewOption.PERFORMANCE, order_by=Screener.OrderBy.PERFORMANCE_YEAR, pages=pages )


def get_rs_ticker_list():
    """"""
    rs_list = []
    for p in range(1,30):
        try:
            performance_table = Screener(filter_options=[Screener.IndexOption.S_AND_P_500],
                                         view_option=Screener.ViewOption.PERFORMANCE,
                                         order_by=Screener.OrderBy.PERFORMANCE_YEAR, pages=[p] )
            rs_table = performance_table.data_frames.get(p)
            rs_list += rs_table['Ticker'].to_list()
        except ValueError:
            break

    print(f"{len(rs_list)} RS Tickers")

    return rs_list


def minervini_method(ticker_list, rs_list):
    """"""

    # for condition_9 of trend_template_screener, it has to compare with S&P500 index
    df_spx = clean_stock_data(yf.download(tickers='^GSPC', period='2y', auto_adjust=True))

    # ticker.info is not used because processing time is too long


    # Create a dataframe to store results later
    radar = pd.DataFrame({
        'Ticker': [],
        'Num_of_contraction': [],
        'Max_contraction': [],
        'Min_contraction': [],
        'Weeks_of_contraction': [],
        'RS_rating': []
    })

    failures = 0

    for ticker_string in ticker_list:
        try:
            ticker_history = clean_stock_data(yf.download(tickers=ticker_string, period='2y', auto_adjust=True))
            trend_template_screener = trend_template(ticker_history) # Determine whether the stocks is in Stage 2
            if trend_template_screener['Pass'].iloc[-1] == 1:
                print(f'{ticker_string} is in Stage 2')
                vcp_screener = list(vcp(ticker_history)) # Determine whether the stocks is in Stage 2
                rs = rs_rating(ticker_string,rs_list) # Calculate RS rating
                if (vcp_screener[-1] == 1) & (rs >= 70):
                    vcp_screener.insert(0,ticker_string)
                    vcp_screener.insert(-1,rs)
                    radar.loc[len(radar)] = vcp_screener[0:6] # Store the results to the dataframe
                    print(f'{ticker_string} has a VCP - rs = {rs}')
                else:
                    print(f'{ticker_string} does not have a VCP')
            else:
                print(f'{ticker_string} is not in Stage 2')
        except:
            failures+=1

    print('Finished!!!')
    print(f'{failures} stocks fail to analyze')


    # In[ ]:


    print(f'{len(radar)} stocks pass')
    # print(radar)


    # In[ ]:
    outdir = datetime.now().isoformat(' ').split(' ')[0]
    os.makedirs(outdir, exist_ok=True)

    for ticker in radar['Ticker']:
        ticker_history = clean_stock_data(yf.download(tickers=ticker, period='2y', auto_adjust=True))
        [local_high, local_low] = local_high_low(ticker_history)
        contraction = contractions(ticker_history,local_high,local_low)
        num_of_contraction = num_of_contractions(contraction)
        local_high = local_high[::-1][0:num_of_contraction]
        local_low = local_low[::-1][0:num_of_contraction]

        plt.plot(range(len(ticker_history.index)),ticker_history['Close'])
        plt.plot(local_high,ticker_history['High'].iloc[local_high],'o')
        plt.plot(local_low,ticker_history['Low'].iloc[local_low],'x')

        plt.title(ticker)
        plt.xlabel('Days')
        plt.ylabel('Close Price')
        #plt.show()
        print(f"Plot {ticker} figure")
        plt.savefig(f"{outdir}/{ticker}.png")
        plt.clf()


    # In[ ]:


    # Define the filename for your Excel file
    # filename = 'C:/Users/marco/Desktop/Trade Resources/Watchlist/vcp_screener.xlsx'
    filename = './vcp_screener.xlsx'

    # Get today's date
    sheetname = dt.today().strftime("%Y_%m_%d")

    # Try to read in the existing file (if it exists)
    try:
        database = pd.read_excel(filename, sheet_name=None)
    except FileNotFoundError:
        database = {}

    # Add today's data to the existing data (if any)
    database[sheetname] = radar

    print("Writing results to new sheet of spreadsheet.")
    # Write the updated data to the Excel file
    with pd.ExcelWriter(filename) as writer:
        for sheet_name, df in database.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def parse_opts():
    """"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-t', dest='newtickers', action='store_true', default=False, help="refresh the ticker list")
    parser.add_argument('-m', dest='minervini', action='store_true', default=False, help="use minervini vcp method")
    parser.add_argument('-g', dest='gain15', action='store_true', default=False, help="use 15% gain method")
    parser.add_argument('-p', dest='percent', type=int, default=15, help="gain percentage")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_opts()
    print(f"{args}")
    #sys.exit()

    if not args.newtickers:
        ticker_list = get_saved_ticker_list()

    if ticker_list == [] or args.newtickers:
        ticker_list = get_screened_tickers()

    if args.minervini:
        minervini_method(ticker_list, get_rs_ticker_list())

    if args.gain15:
        stocks = get_stocks_with_15_percent_gain(ticker_list, args.percent)
        print(f"Stocks with >{args.percent}% gain in any 2-week period:\n")
        for t in stocks:
            for gain in stocks[t]:
                print(f"{gain[1]}")
            print("")
