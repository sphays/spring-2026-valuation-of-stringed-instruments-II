##########################################
# In this file all the additional features that are not part of the auction data but are believed
# to carry predictive power are created.
# These are
# 1. Economic features based on the current stock index', rates', precious metals', currencies' trends, volatilities and sentiments etc.
# 2. Geographical features grouping the city where the instrument was made into broader categrories.
##########################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


###############################
# 1. Economic Feature Generation
###############################

# load data from sales archive and data for currencies conversion (FX data) + inflation adjustment (CPI data)
sales = pd.read_csv('cozio_sales_ALL.csv')
CPI_US = pd.read_csv('./Data/Economic_Data/US_CPI_1982=100_noseasadj.csv', index_col='observation_date', parse_dates=True)
CPI_UK = pd.read_csv('./Data/Economic_Data/UK_CPI_2015=100.csv', index_col='observation_date', parse_dates=True)
CPI_EU = pd.read_csv('./Data/Economic_Data/EUR_CPI_2015=100.csv', index_col='observation_date', parse_dates=True)
FX_GBP = pd.read_csv('./Data/Economic_Data/GBP_USD.csv', index_col='Date', parse_dates=True, dayfirst=True)
FX_GBP = 1/2 * (FX_GBP.ffill() + FX_GBP.bfill()) # fill NaNs with average value of preceding and next 
FX_EUR = pd.read_csv('./Data/Economic_Data/EUR_USD.csv', sep=';', index_col='Date', parse_dates=True)
FX_EUR['Value'] = FX_EUR['Value'].astype(str).str.replace(',', '.', regex=False) # separation of digits with '.', not ','
FX_EUR['Value'] = FX_EUR['Value'].astype(float)

# load Financial Time Series
# 10Y US Gov Bond Yield in percent
US10 = pd.read_csv('./Data/Economic_Data/10Y_Yield_US_percent.csv', index_col='observation_date', parse_dates=True)
US10 = 1/2 * (US10.bfill() + US10.ffill()) # fill with average from previous + next
US10.rename(columns = {'DGS10': '10y_yield_pc'}, inplace=True)
# Gold & SP500 index data: both have previously been preprocessed and inflation-adjusted
Gold = pd.read_csv('./Data/Economic_Data/Gold_real.csv', index_col='date', parse_dates=True)
SP = pd.read_csv('./Data/Economic_Data/SP500_index_real.csv', index_col='Date', parse_dates=True).drop(columns='Close')

# Salse pre-processing: convert time column to pd dateime object
def make_datetime(s):
    '''Time conversion into pd dateimte for cozio_sales_ALL.csv
    
    s is a date string of the form "Mon d, yyyy" or "Mon dd, yyyy"
    here Mon is the 3 letter 'word' abbreviation of each month (e.g. 'Feb' for february),
    d, dd are numbers amd yyyy are numbers (where the number of same letter strings encodes its length)
    '''
    month_literal_number_converter = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    month = month_literal_number_converter[s[:3]]
    year = s[-4:]
    day = s[4:6].replace(',', '')
    if int(day)<10:
        day = '0' + day
    #print(day)
    #print('m', month)
    try:
        return pd.to_datetime(day+'/'+month+'/'+year, format="%d/%m/%Y")
    except:
        return pd.NA # very few false dates cannot be converted -> drop those later as NaNs

####################
# Functions needed for the sales preprocessing (foremost inflation-adjustment in local currencies)
###################
def CPI_preprocess(df):
    #df['observation_date'] = pd.to_datetime(df['observation_date'])
    #df.set_index('observation_date', inplace=True, drop=True)

    df.columns = ['CPI'] # consistent column naming
    # fill NaNs with avg between previous and subsequent value
    if df.CPI.isna().sum() > 0:
        df.loc[df.CPI.isna()] = (0.5 * (df.CPI.ffill() + df.CPI.bfill())).loc[df.CPI.isna()] 
    
    df = year_month_column_maker(df)

    # rebase CPI, such that current month has a multplier of 1, i.e CPI_t -> CPI_t / CPI_last
    df['CPI'] = df['CPI'] / df['CPI'].iloc[-1]

    return df

def currency_preprocess(df, currency, dayfirst=False):
    #df['Date'] = pd.to_datetime(df['Date'], dayfirst=dayfirst)
    #df.set_index('Date', inplace=True, drop=True)
    
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    if currency == 'gbp':
        df = df.rename(columns={'Value': 'gbp_usd'})
    if currency == 'eur':
        df = df.rename(columns={'Value': 'eur_usd'})

    df = year_month_column_maker(df, day=True) # add year, month, day column to adjust prices later

    
    return df


def attach_timeseries_features(sales, ts_df, cols=None):
    """
    Attach latest available values of ts_df to each sale date.

    sales  : DataFrame with DatetimeIndex
    ts_df  : DataFrame with DatetimeIndex
    cols   : columns to merge (default = all)
    """

    if cols:
        ts_df = ts_df[cols]

    return pd.merge_asof(sales.sort_index(), ts_df.sort_index(), left_index=True, right_index=True, direction="backward")



def adjust_prices(df, CPI_US, CPI_UK, CPI_EU, FX_GBP, FX_EUR):

    CPI_US = CPI_US.rename(columns={'CPI': 'cpi_usd'})
    CPI_UK = CPI_UK.rename(columns={'CPI': 'cpi_gbp'})
    CPI_EU = CPI_EU.rename(columns={'CPI': 'cpi_eur'})

    '''#  merge CPI columns into sales df
    df = df.merge(CPI_US, on=['Year','Month'], how='left')
    df = df.merge(CPI_UK, on=['Year','Month'], how='left')
    df = df.merge(CPI_EU, on=['Year','Month'], how='left')

    # merge FX columns into sales df
    df = df.merge(FX_GBP, on=['Year','Month', 'Day'], how='left')
    df = df.merge(FX_EUR, on=['Year','Month', 'Day'], how='left')'''
    df = attach_timeseries_features(df, CPI_US, ['cpi_usd'])
    df = attach_timeseries_features(df, CPI_UK, ['cpi_gbp'])
    df = attach_timeseries_features(df, CPI_EU, ['cpi_eur'])

    df = attach_timeseries_features(df, FX_GBP, ['gbp_usd'])
    df = attach_timeseries_features(df, FX_EUR, ['eur_usd'])

    

    # inflation adjust local currency
    df.loc[df.bold_currency == 'usd', 'usd'] /= df.loc[df.bold_currency == 'usd', 'cpi_usd']
    df.loc[df.bold_currency == 'gbp', 'gbp'] /= df.loc[df.bold_currency == 'gbp', 'cpi_gbp']
    df.loc[df.bold_currency == 'eur', 'eur'] /= df.loc[df.bold_currency == 'eur', 'cpi_eur']

    # USD conversion for EUR and GBP
    df['price_usd_real'] = np.nan

    mask_usd = df.bold_currency == 'usd'
    mask_gbp = df.bold_currency == 'gbp'
    mask_eur = df.bold_currency == 'eur'
    
    df.loc[mask_usd, 'price_usd_real'] = df.loc[mask_usd, 'usd']
    df.loc[mask_gbp, 'price_usd_real'] = df.loc[mask_gbp, 'gbp'] * df.loc[mask_gbp, 'gbp_usd']
    df.loc[mask_eur, 'price_usd_real'] = df.loc[mask_eur, 'eur'] * df.loc[mask_eur, 'eur_usd']


    return df

def year_month_column_maker(df, day=False):
    if df.index.dtype in ['<M8[ns]', 'datetime64[ns]']:
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        if day == True:
            df['Day'] = df.index.day
        return df
    else:
        print('Index datatype must be <M8[ns].')


def sales_preprocessing(df, CPI_US, CPI_UK, CPI_EU, FX_GBP, FX_EUR):

    df.dropna(subset=['sale_date'], inplace=True)
    df.index = df['sale_date']
    df.sort_index(inplace=True)

    df = year_month_column_maker(df, day=True) # add year, month, day column to adjust prices later

    df = adjust_prices(df, CPI_US, CPI_UK, CPI_EU, FX_GBP, FX_EUR)

    return df


#####################################
# Process real data: inflation adjustment from data for 1975 onwards (rest is dropped)
#####################################

# convert to sale date to datetime index
sales['sale_date'] = sales['sale_date'].map(make_datetime) # convert date to pd datetime
sales.dropna(subset=['sale_date'], inplace=True)
sales.index = sales['sale_date']


# make right-formatted CPI series
CPI_US = CPI_preprocess(CPI_US)
CPI_UK = CPI_preprocess(CPI_UK)
CPI_EU = CPI_preprocess(CPI_EU)

# make right-formatted currency conversion series
FX_EUR = currency_preprocess(FX_EUR, 'eur')
FX_GBP = currency_preprocess(FX_GBP, 'gbp', dayfirst=True)

# inflation adjust everything in respective currency
# enhancement of sales data with the economic indicators (FXR, CPI, local conversions to USD)
sales = sales_preprocessing(sales, CPI_US, CPI_UK, CPI_EU, FX_GBP, FX_EUR)

# filters out all those that have no adjusted price, eg those that have bold_currency='nan'
sales.dropna(subset=['price_usd_real'], inplace=True)
sales = sales['1975':] # from 1975 onwards we have most related financial time series (additional features)




##################################
# Enhance data set with economic indicators and create Market Climate Indicator (MCI) as a state-variable for fin. market distress
##################################

########## SP500 Index ########
# returns
# consider minimally monthly returns as art markets are delayed and have frictions and you cannot purchase instantan.
SP['SP500_30d_ret'] = SP['SP500_real'].pct_change(30)
SP['SP500_90d_ret'] = SP['SP500_real'].pct_change(90)
SP['SP500_252d_ret'] = SP['SP500_real'].pct_change(252)

# annualised volatilities on SP500
daily_ret = SP['SP500_real'].pct_change()
SP['SP500_vol_30d']  = daily_ret.rolling(30).std() * np.sqrt(252)
SP['SP500_vol_90d']  = daily_ret.rolling(90).std() * np.sqrt(252)
SP['SP500_vol_252d'] = daily_ret.rolling(252).std() * np.sqrt(252)

# Trend Indicator based on moving averages / smoothing over different time horizons
SP['SP500_ma50']  = SP['SP500_real'].rolling(50).mean()
SP['SP500_ma200'] = SP['SP500_real'].rolling(200).mean()

SP['SP500_trend_ratio'] = SP['SP500_ma50'] / SP['SP500_ma200'] # >1 means bullish


########## FX ########
# FX indicators (based on GBP as it exists for full history unlike Euro)
FX_GBP['gbp_90d_change'] = FX_GBP['gbp_usd'].pct_change(90)
FX_GBP['gbp_vol_30d'] = FX_GBP['gbp_usd'].pct_change().rolling(30).std()
#FX_GBP['gbp_vol_30d'] = 0.5 * (FX_GBP['gbp_vol_30d'].ffill() + FX_GBP['gbp_vol_30d'].bfill()) # linearly interpolate
FX_GBP['gbp_vol_30d'] = FX_GBP['gbp_vol_30d'].ffill()# linearly interpolate

########## Yield ########
# Rate indicators based on 10Y US T-Note Yield level (in percent)
US10['10y_yield_90d_change'] = US10['10y_yield_pc'].diff(90) / 100 # get rid of percent

# all new macroeconomic data
macro = SP.join(FX_GBP).join(US10)
macro.drop(columns=['Year', 'Month', 'Day'], inplace=True)

# z-score normalisation to bring all indicators to the same scale
def z(s, w=252):
    '''
    Rolling Z-score normalisation (zero mean, unit variance) over time window of length w.
    '''
    return (s - s.rolling(w).mean()) / s.rolling(w).std()
    
macro['z_ret'] = z(macro['SP500_252d_ret'])
macro['z_vol'] = z(macro['SP500_vol_30d'])
macro['z_rate'] = z(macro['10y_yield_pc'])
macro['z_drate'] = z(macro['10y_yield_90d_change'])
#macro['z_fx'] = z(macro['gbp_vol_30d'])  # fx vola column
macro['z_fx'] = ((macro['gbp_vol_30d']- macro['gbp_vol_30d'].ffill().rolling(252).mean()) /
                 macro['gbp_vol_30d'].ffill().rolling(252).std())

# construct MARKET CLIMATE INDICATOR (MCI) as single state variable (to reduce noise, multicollinearity, etc.)
# high MCI: strong markets, liquidity, risk appetite. Low MCI: stressed markets
macro['MCI'] = macro['z_ret'] - macro['z_vol'] - macro['z_rate'] - macro['z_drate'] - macro['z_fx']
# smooth indicator to not overfit noise
macro['MCI'] = macro['MCI'].rolling(30, min_periods=1).mean()


################## Add all new relevant features ################
start_idx = macro.dropna().index[0]
# exclude z-score features as they are already in the 'sales' data set 
sales = attach_timeseries_features(sales[start_idx:], macro[start_idx:],
                                   cols = [c for c in macro.columns if c not in ['z_ret', 'z_vol', 'z_rate', 'z_drate', 'z_fx']])

sales = attach_timeseries_features(sales, Gold)

# Save Data
#sales.to_csv('price_adj_all_w_econ_feeatures.csv')
















###############################
# 2. Geographical Feature Generation
###############################
