import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gathering_data import gather_and_fuck_with_it as gf
import warnings
warnings.filterwarnings("ignore")


prices = gf.get_ticker(ticker_list=["BTC-USD"], start = "2000-01-01", end = None, log_returns=False)
prices.columns = ["Close"]
returns = gf.get_ticker(ticker_list=["BTC-USD"], start = "2000-01-01", end = None, log_returns=True)
returns.columns = ["LogReturn"]
df = pd.concat([prices,returns], axis=1)
df = df.iloc[-365:]
df['LogReturn'] = df['LogReturn'].shift(-1)

df['SlowSMA'] = df['Close'].rolling(24).mean()
df['FastSMA'] = df['Close'].rolling(16).mean()

df['Signal'] = np.where(df['FastSMA'] >= df['SlowSMA'], 1, 0)

df['PrevSignal'] = df['Signal'].shift(1)
df['Buy'] = (df['PrevSignal'] == 0) & (df['Signal'] == 1) # Fast < Slow --> Fast > Slow
df['Sell'] = (df['PrevSignal'] == 1) & (df['Signal'] == 0) # Fast > Slow --> Fast < Slow

is_invested = False


def assign_is_invested(row):
    global is_invested
    if is_invested and row['Sell']:
        is_invested = False
    if not is_invested and row['Buy']:
        is_invested = True

    # otherwise, just remain
    return is_invested


df['IsInvested'] = df.apply(assign_is_invested, axis=1)

df['AlgoLogReturn'] = df['IsInvested'] * df['LogReturn']

df['AlgoLogReturn'].sum()

df['LogReturn'].sum()

df['AlgoLogReturn'].std(), df['AlgoLogReturn'].mean()/df['AlgoLogReturn'].std()

df['LogReturn'].std(), df['LogReturn'].mean()/df['LogReturn'].std()

# Start by writing a function to plug in parameters and obtain score
Ntest = 60
def trend_following(df, fast, slow):
  global is_invested
  df['SlowSMA'] = df['Close'].rolling(slow).mean()
  df['FastSMA'] = df['Close'].rolling(fast).mean()
  df['Signal'] = np.where(df['FastSMA'] >= df['SlowSMA'], 1, 0)
  df['PrevSignal'] = df['Signal'].shift(1)
  df['Buy'] = (df['PrevSignal'] == 0) & (df['Signal'] == 1) # Fast < Slow --> Fast > Slow
  df['Sell'] = (df['PrevSignal'] == 1) & (df['Signal'] == 0) # Fast > Slow --> Fast < Slow

  # Split into train and test
  train = df.iloc[:-Ntest]
  test = df.iloc[-Ntest:]

  is_invested = False
  df.loc[:-Ntest,'IsInvested'] = train.apply(assign_is_invested, axis=1)
  df.loc[:-Ntest,'AlgoLogReturn'] = train['IsInvested'] * train['LogReturn']

  is_invested = False
  df.loc[-Ntest:,'IsInvested'] = test.apply(assign_is_invested, axis=1)
  df.loc[-Ntest:,'AlgoLogReturn'] = test['IsInvested'] * test['LogReturn']

  return train['AlgoLogReturn'][:-1].sum(), test['AlgoLogReturn'][:-1].sum()

trend_following(df, 10, 30)

# Let's do a grid search
best_fast = None
best_slow = None
best_score = float('-inf')
for fast in range(3, 10):
  for slow in range(fast + 5, 50):
    score, _ = trend_following(df, fast, slow)
    if score > best_score:
      best_fast = fast
      best_slow = slow
      best_score = score
print(best_fast, best_slow, trend_following(df, best_fast, best_slow))

train = df.iloc[:-Ntest].copy()
test = df.iloc[-Ntest:].copy()

# Total return buy-and-hold train
train['LogReturn'][:-1].sum()

# Total return buy-and-hold test
# Note: last value is NaN, just doing this for the sake of uniformity
test['LogReturn'][:-1].sum()