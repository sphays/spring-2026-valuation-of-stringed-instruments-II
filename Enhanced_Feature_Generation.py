##########################################
# In this file all the additional features that are not part of the auction data but are believed
# to carry predictive power are created.
# These are
# 1. Date, city, country, region of the maker, where possible. 
# 2. Economic features based on the current stock index', rates', precious metals', currencies' trends, volatilities and sentiments etc.

##########################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

###############################
# 1. Geographical and temporal data of the maker.
###############################
import re
import numpy as np
import pandas as pd


# ── transfo: string date → approximate numeric year ───────────────────────────

CENTURY_MIDPOINTS = {
    'early': -25, 'mid': 0, 'late': 25,   # offsets from century midpoint (50)
}

def transfo(date_str):
    """
    Convert a messy string date (fl., b., d., c., ranges, centuries) to an
    approximate numeric year (float). Returns np.nan if unparseable.

    Examples:
        "1750"              → 1750
        "1680–1732"         → 1706
        "fl. 1868–"         → 1868
        "fl. 1868–1903"     → 1886
        "c. 1901"           → 1901
        "b. 1971"           → 1971
        "d. 1850"           → 1850
        "fl. 19th c."       → 1850
        "early 19th c."     → 1810
        "mid 18th c."       → 1750
        "late 20th c."      → 1975
        "fl. early 19th c." → 1810
        "from 1922"         → 1922
        "1835"              → 1835
    """
    if not date_str or pd.isna(date_str):
        return np.nan

    s = str(date_str).strip()

    # ── strip fl., b., d., c. prefixes ──────────────────────────────────────
    s_clean = re.sub(r'^(fl\.|b\.|d\.|c\.)\s*', '', s, flags=re.I).strip()

    # ── "from YYYY" ──────────────────────────────────────────────────────────
    m = re.match(r'^from\s+(\d{4})', s_clean, re.I)
    if m:
        return float(m.group(1))

    # ── range "YYYY–YYYY" or "YYYY-YYYY" ────────────────────────────────────
    m = re.match(r'^(\d{4})\s*[-–]\s*(\d{4})', s_clean)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2

    # ── open range "YYYY–" ───────────────────────────────────────────────────
    m = re.match(r'^(\d{4})\s*[-–]\s*$', s_clean)
    if m:
        return float(m.group(1))

    # ── single year "YYYY" ───────────────────────────────────────────────────
    m = re.match(r'^(\d{4})$', s_clean)
    if m:
        return float(m.group(1))

    # ── century expressions: "early/mid/late Nth c." ─────────────────────────
    m = re.search(r'(early|mid|late)?\s*(\d+)(st|nd|rd|th)\s*c', s_clean, re.I)
    if m:
        qualifier = (m.group(1) or 'mid').lower()
        century   = int(m.group(2))
        base      = (century - 1) * 100  # e.g. 19th c. → 1800
        offsets   = {'early': 15, 'mid': 50, 'late': 80}
        return float(base + offsets[qualifier])

    # ── multiple values separated by "/" (e.g. firm dates) ──────────────────
    parts = re.findall(r'\d{4}', s)
    if parts:
        years = [float(y) for y in parts]
        return sum(years) / len(years)

    return np.nan


# ── imputer ───────────────────────────────────────────────────────────────────

def imputer(sales: pd.DataFrame,
            makers: pd.DataFrame,
            instruments: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing city_maker and make_date in sales, using three fallback
    sources for each field.

    city_maker fallback chain (per maker_id):
      1. Most frequent city_maker across all sales of this maker
      2. Most frequent city_maker across instruments of this maker
      3. city column in makers df for this maker_id

    make_date fallback chain (per maker_id):
      1. Mean make_date across instruments of this maker (numeric)
      2. transfo(date) from makers df for this maker_id

    Parameters
    ----------
    sales       : sales DataFrame; must have columns maker_id, city_maker.
                  make_date column is created if absent.
    makers      : makers DataFrame; index or column maker_id, columns city, date.
    instruments : instruments DataFrame; must have columns maker_id, city_maker.
                  Optionally make_date.

    Returns
    -------
    sales DataFrame with city_maker and make_date filled where possible.
    """
    sales = sales.copy()

    # Ensure make_date column exists
    if 'make_date' not in sales.columns:
        sales['make_date'] = np.nan

    # ── Normalise makers index ────────────────────────────────────────────────
    if 'maker_id' in makers.columns:
        makers_idx = makers.set_index('maker_id')
    else:
        makers_idx = makers.copy()

    # ── Pre-compute city lookups ──────────────────────────────────────────────

    def _most_frequent(series):
        """Return most frequent non-null value, or None."""
        s = series.dropna()
        s = s[s.str.strip() != ''] if s.dtype == object else s
        return s.mode().iloc[0] if len(s) > 0 else None

    # 1. Most frequent city in sales per maker_id
    city_from_sales = (
        sales.groupby('maker_id')['city_maker']
        .agg(_most_frequent)
        .rename('city_sales')
    )

    # 2. Most frequent city in instruments per maker_id
    if 'city_maker' in instruments.columns:
        city_from_instruments = (
            instruments.groupby('maker_id')['city_maker']
            .agg(_most_frequent)
            .rename('city_instruments')
        )
    else:
        city_from_instruments = pd.Series(dtype=str, name='city_instruments')

    # 3. City from makers df
    city_from_makers = (
        makers_idx['city'].dropna() if 'city' in makers_idx.columns
        else pd.Series(dtype=str)
    )

    # ── Pre-compute make_date lookups ─────────────────────────────────────────

    # 1. Mean make_date from instruments per maker_id
    if 'make_date' in instruments.columns:
        make_date_from_instruments = (
            instruments.groupby('maker_id')['make_date']
            .apply(lambda s: pd.to_numeric(s, errors='coerce').mean())
            .rename('make_date_instruments')
        )
    else:
        make_date_from_instruments = pd.Series(dtype=float, name='make_date_instruments')

    # 2. transfo(date) from makers df
    if 'date' in makers_idx.columns:
        make_date_from_makers = makers_idx['date'].apply(transfo).rename('make_date_makers')
    else:
        make_date_from_makers = pd.Series(dtype=float, name='make_date_makers')

    # ── Apply imputation row by row (vectorised per maker_id) ─────────────────

    def _fill_city(row):
        if pd.notna(row['city_maker']) and str(row['city_maker']).strip() not in ('', 'nan'):
            return row['city_maker']
        mid = row['maker_id']
        # 1. sales
        v = city_from_sales.get(mid)
        if pd.notna(v) and str(v).strip() not in ('', 'nan'):
            return v
        # 2. instruments
        v = city_from_instruments.get(mid)
        if pd.notna(v) and str(v).strip() not in ('', 'nan'):
            return v
        # 3. makers
        try:
            v = city_from_makers.loc[mid]
            if pd.notna(v) and str(v).strip() not in ('', 'nan'):
                return v
        except KeyError:
            pass
        return row['city_maker']

    def _fill_make_date(row):
        if pd.notna(row['make_date']):
            return row['make_date']
        mid = row['maker_id']
        # 1. instruments mean
        try:
            v = make_date_from_instruments.loc[mid]
            if pd.notna(v):
                return v
        except KeyError:
            pass
        # 2. makers date string
        try:
            v = make_date_from_makers.loc[mid]
            if pd.notna(v):
                return v
        except KeyError:
            pass
        return row['make_date']

    sales['city_maker'] = sales.apply(_fill_city,     axis=1)
    sales['make_date']  = sales.apply(_fill_make_date, axis=1)
                
    city_map=pd.read_csv('./Data/Geo_Data/city_map.csv',index='city_maker')
                
    for col in [
    "country_iso1",
            "admin1_name",
     "admin2_name",
        ]:
        sales[col] = sales["city_maker"].map(city_map[col])
    
    return sales

sales = pd.read_csv('.Data/tarisio_data/cozio_sales_ALL.csv')
instruments=pd.read_csv('.Data/tarisio_data/instruments_scraping_complete.csv',index_col='instrument_id')
makers=pd.read_csv('.Data/tarisio_data/makers.csv',index_col='maker_id')

sales = imputer(sales, makers, instruments)
###############################
# 2. Economic Feature Generation
###############################

# load data from sales archive and data for currencies conversion (FX data) + inflation adjustment (CPI data)
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

################## 2: Geographical Feature Generation ################
# ----------------------------- Normalization -----------------------------

CHAR_MAP = str.maketrans({
    "ø": "o", "Ø": "O",
    "ł": "l", "Ł": "L",
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th",
    "æ": "ae", "Æ": "Ae",
    "œ": "oe", "Œ": "Oe",
})

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.translate(CHAR_MAP)
    s = strip_accents(s)
    s = s.casefold().strip()
    s = re.sub(r"[’`]", "'", s)
    s = re.sub(r"[\s\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ----------------------------- Parsing helpers -----------------------------
US_STATE_ABBR = set("""
AL AK AZ AR CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN
MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT
WA WI WV WY DC
""".split())

US_STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

def parse_place(raw: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Returns: (place_core, subregion_hint, state_hint, country_hint)

    - "City, ST" -> state_hint=ST, country_hint='US'
    - "City, StateName" -> state_hint=abbr, country_hint='US'
    - "Name (Qualifier)" -> place_core=Name, subregion_hint=Qualifier
    - Non-US comma suffix like "Worthing, Sussex" -> subregion_hint="Sussex"
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ("", None, None, None)

    s = str(raw).strip()
    if not s:
        return ("", None, None, None)

    m_paren = re.match(r"^(.*?)\s*\((.*?)\)\s*$", s)
    if m_paren:
        core = m_paren.group(1).strip()
        qual = m_paren.group(2).strip()
        return (core, qual or None, None, None)

    if "," in s:
        left, right = s.rsplit(",", 1)
        left = left.strip()
        right_stripped = right.strip()

        m_abbr = re.match(r"^([A-Za-z]{2})$", right_stripped)
        if m_abbr:
            st = m_abbr.group(1).upper()
            if st in US_STATE_ABBR:
                return (left, None, st, "US")

        right_norm = norm_text(right_stripped)
        if right_norm in US_STATE_NAME_TO_ABBR:
            st = US_STATE_NAME_TO_ABBR[right_norm]
            return (left, None, st, "US")

        # treat as subregion hint (e.g. Sussex, Haute-Saone, etc.)
        return (left, right_stripped or None, None, None)

    return (s, None, None, None)


# ----------------------------- GeoNames loading -----------------------------
@dataclass(frozen=True)
class Candidate:
    geonameid: str
    name: str
    asciiname: str
    country: str
    admin1: str
    admin2: str
    feature_class: str
    feature_code: str
    population: int

def load_admin1(admin1_path: str) -> Dict[str, str]:
    m = {}
    with open(admin1_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip():
                m[parts[0].strip()] = parts[1].strip()
    return m

def load_admin2(admin2_path: str) -> Dict[str, str]:
    m = {}
    with open(admin2_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip():
                m[parts[0].strip()] = parts[1].strip()
    return m

def build_candidates_index(
    geonames_path: str,
    target_keys: set,
    include_admin_features: bool = True,
) -> Dict[str, List[Candidate]]:
    allowed_A_codes = {"ADM1", "ADM2", "ADM3", "ADM4", "ADMD", "ADM5"}
    cand_by_key: Dict[str, List[Candidate]] = {k: [] for k in target_keys}

    def maybe_add(key: str, parts: List[str]) -> None:
        fclass = parts[6]
        fcode = parts[7]
        if fclass == "P":
            pass
        elif include_admin_features and fclass == "A" and fcode in allowed_A_codes:
            pass
        else:
            return

        try:
            pop = int(parts[14]) if parts[14] else 0
        except ValueError:
            pop = 0

        cand_by_key[key].append(
            Candidate(
                geonameid=parts[0],
                name=parts[1],
                asciiname=parts[2],
                country=parts[8],
                admin1=parts[10],
                admin2=parts[11],
                feature_class=fclass,
                feature_code=fcode,
                population=pop,
            )
        )

    # pass 1: name/asciiname
    with open(geonames_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue

            k_ascii = norm_text(parts[2])
            if k_ascii in target_keys:
                maybe_add(k_ascii, parts)

            k_name = norm_text(parts[1])
            if k_name in target_keys and k_name != k_ascii:
                maybe_add(k_name, parts)

    unmatched = {k for k, v in cand_by_key.items() if len(v) == 0}
    if not unmatched:
        return cand_by_key

    # pass 2: alternatenames for remaining unmatched keys
    with open(geonames_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            alts = parts[3]
            if not alts or not unmatched:
                continue

            for alt in alts.split(","):
                k = norm_text(alt)
                if k in unmatched:
                    maybe_add(k, parts)

    return cand_by_key


# ----------------------------- Overrides-----------------------------
#override specs for ambiguous place names that were determined to not be the one with the highest score but rather another one based on manual inspection. The keys are the normalized place names and the values are dicts with the following possible keys:
# - country: ISO country code to filter candidates by country   
# - lookup: list of normalized place names to look for in the candidate's name/asciiname/alternatenames (in order of preference)
OVERRIDES: Dict[str, Dict[str, Union[str, List[str], bool, List[str]]]] = {
    "gerona":      {"country": "IT", "lookup": ["fosso gerona"]},
    "mantua":      {"country": "IT", "lookup": ["mantova", "mantua"], "prefer_class": "A", "prefer_codes": ["ADM2", "ADM1"]},
    "cheshire":    {"country": "GB", "prefer_class": "A"},
    "padua":       {"country": "IT", "lookup": ["padova", "padua"]},
    "beaconsfield":{"country": "GB"},
    "lugo":        {"country": "IT"},
    "san marino":  {"country": "SM"},
    "devonport":   {"country": "GB"},
    "tournay":     {"country": "BE", "lookup": ["tournai", "tournay"]},
    "berne":       {"country": "CH", "lookup": ["bern", "berne"]},
    "antwerp":     {"country": "BE"},
    "valencia":    {"country": "ES"},
    "kew":         {"country": "GB"},
    "gand":        {"country": "BE", "lookup": ["gent", "ghent", "gand"]},
    "salo":        {"country": "IT"},
    "surrey":      {"country": "GB", "prefer_class": "A"},
    "hanwell":     {"country": "GB"},
    "kensington":  {"country": "GB"},
    "hull":        {"country": "GB"},
    "bernal":      {"country": "AR"},
    "moravia":     {"country": "CZ", "prefer_class": "A"},
    "orleans":     {"country": "FR"},
    "aquila":      {"country": "IT", "lookup": ["l'aquila", "laquila", "l aquila", "aquila"]},
    "leith":       {"country": "GB"},
    "pesth":       {"country": "HU", "lookup": ["pest", "pesth"]},
    "leonardo":    {"country": "IT"},
    "geneva":      {"country": "CH"},
    "san remo":    {"country": "IT", "lookup": ["sanremo", "san remo"]},
    "cumberland":  {"country": "GB", "prefer_class": "A"},
    "lancaster":   {"country": "US", "state_hint": "PA"},
    "alva":        {"country": "GB"},
    "hanley":      {"country": "GB"},
    "franklin":    {"blank": True},
}

# ----------------------------- Main enrichment -----------------------------
def enrich_city_maker(
    df: pd.DataFrame,
    city_col: str = "city_maker",
    geonames_path: str = "allCountries.txt",
    admin1_path: str = "admin1CodesASCII.txt",
    admin2_path: str = "admin2Codes.txt",
    print_ambiguous: bool = True,
    top_k_print: int = 5,
    use_subregion_hint: bool = True,
    hint_bonus: float = 4.0,
) -> pd.DataFrame:
    """
    Adds:
      - country_iso1
      - admin1_name
      - admin2_name
      - candidate_count
      - is_ambiguous

    Uses subregion hint from parentheses or non-US comma suffix to filter/boost candidates.
    """
    out = df.copy()
    if city_col not in out.columns:
        raise ValueError(f"Column '{city_col}' not found.")

    # Prepare unique places
    uniques = pd.Series(out[city_col].dropna().unique())
    parsed = []
    for raw in uniques.tolist():
        core, sub_hint, st_hint, c_hint = parse_place(raw)
        core_key = norm_text(core)
        if core_key:
            parsed.append((raw, core, core_key, sub_hint, st_hint, c_hint))

    if not parsed:
        out["country_iso1"] = pd.NA
        out["admin1_name"] = pd.NA
        out["admin2_name"] = pd.NA
        out["candidate_count"] = pd.NA
        out["is_ambiguous"] = False
        return out

    places = pd.DataFrame(parsed, columns=["place_raw", "place_core", "place_key", "subregion_hint", "state_hint", "country_hint"])

    # Build target keys (include override lookup keys)
    target_keys = set(places["place_key"].unique())
    for _, spec in OVERRIDES.items():
        lookup = spec.get("lookup", None)
        if isinstance(lookup, str):
            target_keys.add(norm_text(lookup))
        elif isinstance(lookup, list):
            for name in lookup:
                target_keys.add(norm_text(name))

    admin1_map = load_admin1(admin1_path)
    admin2_map = load_admin2(admin2_path)
    cand_by_key = build_candidates_index(geonames_path, target_keys, include_admin_features=True)

    def admin_names_for(c: Candidate) -> Tuple[str, str]:
        a1 = admin1_map.get(f"{c.country}.{c.admin1}", "") if c.country and c.admin1 else ""
        a2 = admin2_map.get(f"{c.country}.{c.admin1}.{c.admin2}", "") if c.country and c.admin1 and c.admin2 else ""
        return a1, a2

    def hint_matches(c: Candidate, hint_norm: str) -> bool:
        if not hint_norm:
            return False
        a1, a2 = admin_names_for(c)
        return (hint_norm in norm_text(a1)) or (hint_norm in norm_text(a2))

    def score_candidate(c: Candidate, prefer_class: Optional[str], prefer_codes: Optional[List[str]], hint_norm: str) -> float:
        s = math.log1p(max(c.population, 0))
        if c.feature_class == "P":
            s += 2.0
        if prefer_class and c.feature_class == prefer_class:
            s += 3.0
        if prefer_codes and c.feature_code in set(prefer_codes):
            s += 2.0
        if use_subregion_hint and hint_norm and hint_matches(c, hint_norm):
            s += hint_bonus
        return s

    resolved_rows = []
    ambiguous_reports = []

    for _, r in places.iterrows():
        raw = r["place_raw"]
        core_key = r["place_key"]
        hint_norm = norm_text(r["subregion_hint"]) if (use_subregion_hint and r["subregion_hint"]) else ""

        spec = OVERRIDES.get(core_key, {})
        if spec.get("blank", False):
            resolved_rows.append({
                "place_raw": raw,
                "country_iso1": pd.NA,
                "admin1_name": pd.NA,
                "admin2_name": pd.NA,
                "candidate_count": 0,
                "is_ambiguous": False,
            })
            continue

        forced_country = spec.get("country", None) or r["country_hint"]
        forced_state   = spec.get("state_hint", None) or r["state_hint"]
        prefer_class   = spec.get("prefer_class", None)
        prefer_codes   = spec.get("prefer_codes", None)

        lookup_names = spec.get("lookup", None)
        if isinstance(lookup_names, str):
            lookup_keys = [norm_text(lookup_names)]
        elif isinstance(lookup_names, list):
            lookup_keys = [norm_text(x) for x in lookup_names]
        else:
            lookup_keys = [core_key]

        # gather candidates 
        candidates = []
        seen = set()
        for lk in lookup_keys:
            for c in cand_by_key.get(lk, []):
                if c.geonameid not in seen:
                    seen.add(c.geonameid)
                    candidates.append(c)

        # forced country filter (hard)
        if forced_country:
            candidates = [c for c in candidates if c.country == forced_country]

        # forced US state filter when possible (hard)
        if forced_state:
            candidates2 = [c for c in candidates if c.country == "US" and c.admin1 == forced_state]
            if candidates2:
                candidates = candidates2

        # if forced country yields nothing: blank 
        if forced_country and not candidates:
            resolved_rows.append({
                "place_raw": raw,
                "country_iso1": pd.NA,
                "admin1_name": pd.NA,
                "admin2_name": pd.NA,
                "candidate_count": 0,
                "is_ambiguous": False,
            })
            continue

        # subregion hint narrowing
        if use_subregion_hint and hint_norm and len(candidates) > 1:
            matching = [c for c in candidates if hint_matches(c, hint_norm)]
            if matching:
                candidates = matching  # only narrow if it actually helps

        cand_count = len(candidates)
        is_amb = cand_count > 1

        if candidates:
            scored = [(score_candidate(c, prefer_class, prefer_codes, hint_norm), c) for c in candidates]
            scored.sort(key=lambda t: t[0], reverse=True)
            best = scored[0][1]

            country_iso1 = best.country or pd.NA
            a1_key = f"{best.country}.{best.admin1}" if best.country and best.admin1 else None
            admin1_name = admin1_map.get(a1_key, pd.NA) if a1_key else pd.NA
            a2_key = f"{best.country}.{best.admin1}.{best.admin2}" if best.country and best.admin1 and best.admin2 else None
            admin2_name = admin2_map.get(a2_key, pd.NA) if a2_key else pd.NA

            # Print remaining ambiguous cases (excluding overrides)
            if print_ambiguous and is_amb and core_key not in OVERRIDES:
                top = scored[:top_k_print]
                opts = []
                for s_val, c in top:
                    a1, a2 = admin_names_for(c)
                    opts.append((s_val, c.name, c.asciiname, c.country, a1, a2, c.feature_class, c.feature_code, c.population))
                ambiguous_reports.append((raw, cand_count, r["subregion_hint"], opts))
        else:
            country_iso1 = pd.NA
            admin1_name = pd.NA
            admin2_name = pd.NA

        resolved_rows.append({
            "place_raw": raw,
            "country_iso1": country_iso1,
            "admin1_name": admin1_name,
            "admin2_name": admin2_name,
            "candidate_count": cand_count,
            "is_ambiguous": bool(is_amb),
        })

    mapping = pd.DataFrame(resolved_rows).drop_duplicates("place_raw")
    out = out.merge(mapping, left_on=city_col, right_on="place_raw", how="left").drop(columns=["place_raw"])
    out["is_ambiguous"] = out["is_ambiguous"].fillna(False).astype(bool)

    if print_ambiguous and ambiguous_reports:
        print("\n--- Ambiguous place names (not covered by overrides) ---")
        for raw, n, hint, opts in ambiguous_reports:
            hint_txt = f" | hint='{hint}'" if hint else ""
            print(f"\nAMBIGUOUS: '{raw}' (candidates={n}{hint_txt})")
            for s_val, name, asciiname, cc, a1, a2, fcl, fcd, pop in opts:
                a1_txt = f" | admin1={a1}" if a1 else ""
                a2_txt = f" | admin2={a2}" if a2 else ""
                print(f"  score={s_val:0.3f} | {name} ({asciiname}) | {cc}{a1_txt}{a2_txt} | {fcl}/{fcd} | pop={pop}")

    return out


sales = enrich_city_maker(sales, city_col="city_maker",
                        geonames_path="Data/Geo_Data/allCountries.txt",
                        admin1_path="Data/Geo_Data/admin1CodesASCII.txt",
                        admin2_path="Data/Geo_Data/admin2Codes.txt",
                        print_ambiguous=False)

# sales.to_csv("price_adj_all_w_econ_and_geo_features.csv", index=False)
sales.head(50)

################# 3: Merge with maker data ##############################
def merge_maker_data(
    sales_df,
    maker_df,
    sales_maker_col="maker_name",
    maker_maker_col="maker_name",
    how="left",
    suffixes=("", "_maker"),
    normalize_names=True,
    verbose=True,
):
    """
    Merge maker-level data into sales-level data by maker name.
    If maker_df has multiple rows with the same maker name, this function keeps
    ONLY the first occurrence (in maker_df order) and discards the rest.
    Also drops the 'info' column from maker_df before merging (if present).
    """
    import pandas as pd
    import re

    s = sales_df.copy()
    m = maker_df.copy()

    if sales_maker_col not in s.columns:
        raise ValueError(f"'{sales_maker_col}' not found in sales_df")
    if maker_maker_col not in m.columns:
        raise ValueError(f"'{maker_maker_col}' not found in maker_df")

    # Drop maker 'info' column if present
    m = m.drop(columns=["info"], errors="ignore")

    def _norm(x):
        if pd.isna(x):
            return pd.NA
        x = str(x).strip().casefold()
        x = re.sub(r"\s+", " ", x)
        return x

    if normalize_names:
        s["_maker_key"] = s[sales_maker_col].map(_norm)
        m["_maker_key"] = m[maker_maker_col].map(_norm)
        key = "_maker_key"
    else:
        key = maker_maker_col
        s["_maker_key"] = s[sales_maker_col]
        m["_maker_key"] = m[maker_maker_col]

    # Keep first maker row per key (preserve maker_df order)
    if verbose:
        dup_counts = m["_maker_key"].value_counts(dropna=False)
        dups = dup_counts[dup_counts > 1]
        if len(dups):
            print("WARNING: maker_df has duplicate maker names. Keeping ONLY the first row for each name.")
            print(dups.head(20))

    m_first = m.drop_duplicates(subset=["_maker_key"], keep="first")

    merged = s.merge(
        m_first,
        left_on="_maker_key",
        right_on="_maker_key",
        how=how,
        suffixes=suffixes,
    )

    if verbose:
        maker_cols_added = [c for c in m_first.columns if c not in {maker_maker_col, "_maker_key"}]
        if maker_cols_added:
            matched = merged[maker_cols_added].notna().any(axis=1).sum()
            print(f"Merged maker data: matched {matched}/{len(merged)} sales rows.")
            if matched < len(merged):
                unmatched = merged.loc[~merged[maker_cols_added].notna().any(axis=1), sales_maker_col].dropna()
                if len(unmatched):
                    print("Top unmatched maker_name values (sales_df):")
                    print(unmatched.value_counts().head(15))

    # Drop temp key
    merged = merged.drop(columns=["_maker_key"], errors="ignore")

    return merged

maker_df = pd.read_csv("maker_profiles_engineered.csv") 

# Rename duplicate column labels by appending _1, _2, etc 
cols = list(sales.columns)
seen = {}
new_cols = []
for c in cols:
    if c not in seen:
        seen[c] = 0
        new_cols.append(c)
    else:
        seen[c] += 1
        new_cols.append(f"{c}_{seen[c]}")
sales.columns = new_cols

sales = merge_maker_data(
    sales,
    maker_df,
    sales_maker_col="maker_name",
    maker_maker_col="maker_name",
    normalize_names=True,
    verbose=True,
)


######################### Save Data #########################
sales.to_csv("price_adj_w_all_features.csv", index=False)
