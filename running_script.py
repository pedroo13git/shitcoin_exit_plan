from gathering_data import gather_and_fuck_with_it as gf
from make_me_money import modern_portfolio as mp

prices = gf.get_ticker(ticker_list=["BTC-USD", "ETH-USD"], start = "2000-01-01", end = None, log_returns=True)

mean_return, cov_returns = gf.historical_mean_and_cov(prices)

mp.make_me_a_portfolio_please(mean_return, cov_returns)
