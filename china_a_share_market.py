import pandas as pd
from meta import config
from meta.data_processors.tushare import Tushare
from meta.env_stock_trading.env_stocktrading_China_A_shares import StockTradingEnv
# from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from agents.stablebaselines3_models import DRLAgent, DRLEnsembleAgent
import numpy as np
import tushare
import os
from matplotlib import pyplot as plt
pd.options.display.max_columns = None

print("ALL Modules have been imported!")

"""### Create folders"""


if not os.path.exists("./datasets" ):
    os.makedirs("./datasets" )
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models" )
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log" )
if not os.path.exists("./results" ):
    os.makedirs("./results" )

"""### Download data, cleaning and feature engineering"""

ticket_list=['600000.SH', '600009.SH', '600016.SH', '600028.SH', '600030.SH',
       '600031.SH', '600036.SH', '600050.SH', '600104.SH', '600196.SH',
       '600276.SH', '600309.SH', '600519.SH', '600547.SH', '600570.SH']
# ticket_list=['600000.SH', '600009.SH', '600016.SH', '600028.SH', '600030.SH',
#        '600031.SH', '600036.SH', '600048.SH', '600050.SH', '600104.SH',
#        '600196.SH', '600276.SH', '600309.SH', '600519.SH', '600547.SH',
#        '600570.SH', '600570.SH', '600585.SH', '600588.SH', '600690.SH',
#        '600709.SH', '600745.SH', '600809.SH', '600837.SH', '600887.SH',
#        '600893.SH', '600918.SH', '601012.SH', '601066.SH', '601088.SH',
#        '601138.SH', '601166.SH', '601211.SH', '601288.SH', '601318.SH',
#        '601366.SH', '601398.SH', '601601.SH', '601628.SH', '601668.SH',
#        '601688.SH', '601818.SH', '601857.SH', '601888.SH', '601899.SH',
#        '601995.SH', '603259.SH', '603288.SH', '603501.SH', '603986.SH']
train_start_date='2007-01-01'
train_stop_date='2019-01-01'
val_start_date='2019-01-01'
val_stop_date='2022-09-30'

# token='27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5'
token='9527659cc379610d56240584d91421a39d34954f728db2ebd0623479'

# download and clean
ts_processor = Tushare(data_source="tushare",
                                   start_date=train_start_date,
                                   end_date=val_stop_date,
                                   time_interval="1d",
                                   token=token)
ts_processor.download_data(ticker_list=ticket_list)

ts_processor.clean_data()

# add_technical_indicator
ts_processor.add_technical_indicator(config.INDICATORS)
ts_processor.clean_data()

#add turbulence

def calculate_turbulence(data):
  """calculate turbulence index based on dow 30"""
  # can add other market assets
  df = data.copy()
  df_price_pivot = df.pivot(index="date", columns="tic", values="close")
  # use returns to calculate turbulence
  df_price_pivot = df_price_pivot.pct_change()

  unique_date = df.date.unique()
  # start after a year
  start = 252
  turbulence_index = [0] * start
  # turbulence_index = [0]
  count = 0
  for i in range(start, len(unique_date)):
      current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
      # use one year rolling window to calcualte covariance
      hist_price = df_price_pivot[
          (df_price_pivot.index < unique_date[i])
          & (df_price_pivot.index >= unique_date[i - 252])
      ]
      # Drop tickers which has number missing values more than the "oldest" ticker
      filtered_hist_price = hist_price.iloc[
          hist_price.isna().sum().min() :
      ].dropna(axis=1)

      cov_temp = filtered_hist_price.cov()
      current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
          filtered_hist_price, axis=0
      )
      # cov_temp = hist_price.cov()
      # current_temp=(current_price - np.mean(hist_price,axis=0))

      temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
          current_temp.values.T
      )
      if temp > 0:
          count += 1
          if count > 2:
              turbulence_temp = temp[0][0]
          else:
              # avoid large outlier because of the calculation just begins
              turbulence_temp = 0
      else:
          turbulence_temp = 0
      turbulence_index.append(turbulence_temp)
  try:
      turbulence_index = pd.DataFrame(
          {"date": df_price_pivot.index, "turbulence": turbulence_index}
      )
  except ValueError:
      raise Exception("Turbulence information could not be added.")
  return turbulence_index

def add_turbulence(data):
  """
  add turbulence index from a precalcualted dataframe
  :param data: (df) pandas dataframe
  :return: (df) pandas dataframe
  """
  df = data.copy()
  turbulence_index = calculate_turbulence(df)
  df = df.merge(turbulence_index, on="date")
  df = df.sort_values(["date", "tic"]).reset_index(drop=True)
  return df

ts_processor.dataframe = add_turbulence(ts_processor.dataframe)

ts_processor.clean_data()

"""### Split traning dataset"""

train = ts_processor.data_split(ts_processor.dataframe, train_start_date, train_stop_date)

stock_dimension = len(train.tic.unique())
state_space = stock_dimension*(len(config.INDICATORS)+2)+1
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

"""## Train"""

env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 1000,
    "initial_amount": 1000000,
    "buy_cost_pct":6.87e-5,
    "sell_cost_pct":1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy":True,
    "hundred_each_trade":True
}
# TIME_STEPS = 10_000

e_train_gym = StockTradingEnv(df = train, **env_kwargs)



env_train, _ = e_train_gym.get_sb_env()
agent = DRLAgent(env = env_train)

for TIME_STEPS in np.logspace(0, 6, num=7):
    """### DDPG"""
    # agent = DRLAgent(env = env_train)
    # DDPG_PARAMS = {
    #                 "batch_size": 256,
    #                "buffer_size": 50000,
    #                "learning_rate": 0.0005,
    #                "action_noise":"normal",
    #                 }
    # POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
    # model_td3 = agent.get_model("ddpg", model_kwargs = DDPG_PARAMS, policy_kwargs=POLICY_KWARGS)
    # model_td3 = agent.get_model("ddpg")
    # trained_td3 = agent.train_model(model=model_td3,
    #                               tb_log_name='ddpg',
    #                               total_timesteps=TIME_STEPS)
    """### TD3"""

    model_td3 = agent.get_model("td3")
    trained_td3 = agent.train_model(model=model_td3, tb_log_name='td3', total_timesteps=TIME_STEPS)


    """### A2C"""

    model_a2c = agent.get_model("a2c")

    # Commented out IPython magic to ensure Python compatibility.
    # %%capture
    trained_a2c = agent.train_model(model=model_a2c,
                                 tb_log_name='a2c',
                                 total_timesteps=TIME_STEPS)

    # """### PPO"""
    #
    # model_sac = agent.get_model("ppo")
    # trained_sac = agent.train_model(model=model_sac,
    #                              tb_log_name='ppo',
    #                              total_timesteps=TIME_STEPS)
    #
    """### SAC"""

    model_sac = agent.get_model("sac")
    trained_sac = agent.train_model(model=model_sac,
                                 tb_log_name='sac',
                                 total_timesteps=TIME_STEPS)



    """### Ensembled Algorithms"""
    env_kwargs = {
        "stock_dim": stock_dimension,
        "hmax": 1000,
        "initial_amount": 1000000,
        "buy_cost_pct":6.87e-5,
        "sell_cost_pct":1.0687e-3,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "print_verbosity": 1,
    }
    rebalance_window = 63 # rebalance_window is the number of days to retrain the model
    validation_window = 63 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

    ensemble_agent = DRLEnsembleAgent(df=ts_processor.dataframe,
                     train_period=(train_start_date,train_stop_date),
                     val_test_period=(val_start_date,val_stop_date),
                     rebalance_window=rebalance_window,
                     validation_window=validation_window,
                     **env_kwargs)

    # A2C_model_kwargs = {
    #             'n_steps': 5,
    #             'ent_coef': 0.005,
    #             'learning_rate': 0.0007
    #             }
    #
    # PPO_model_kwargs = {
    #             "ent_coef":0.01,
    #             "n_steps": 2048,
    #             "learning_rate": 0.00025,
    #             "batch_size": 128
    #             }
    #
    # DDPG_model_kwargs = {
    #             #"action_noise":"ornstein_uhlenbeck",
    #             "buffer_size": 10_000,
    #             "learning_rate": 0.0005,
    #             "batch_size": 64
    #           }

    timesteps_dict = {'a2c' : TIME_STEPS,
              'sac' : TIME_STEPS,
              'td3' : TIME_STEPS
              }

    ensemble_df = ensemble_agent.run_ensemble_strategy(timesteps_dict)

    processed = ts_processor.dataframe
    unique_trade_date = processed[(processed.date >= val_start_date)&(processed.date < val_stop_date)].date.unique()

    df_trade_date = pd.DataFrame({'datadate':unique_trade_date})

    df_account_value=pd.DataFrame()
    for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
        temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format('ensemble',i))
        df_account_value = df_account_value.append(temp,ignore_index=True)
    sharpe=(252**0.5)*df_account_value.account_value.pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
    print('Sharpe Ratio: ',sharpe)

    """## Trade"""

    trade = ts_processor.data_split(ts_processor.dataframe, val_start_date, val_stop_date)
    env_kwargs = {
        "stock_dim": stock_dimension,
        "hmax": 1000,
        "initial_amount": 1000000,
        "buy_cost_pct":6.87e-5,
        "sell_cost_pct":1.0687e-3,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "print_verbosity": 1,
        "initial_buy":False,
        "hundred_each_trade":True
    }
    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)

    # df_account_value_ensemble, df_actions_ensemble = DRLAgent.DRL_prediction(model=trained_ensemble,
    #                        environment = e_trade_gym)

    df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(model=trained_td3,
                           environment = e_trade_gym)

    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(model=trained_a2c,
                           environment = e_trade_gym)

    df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(model=trained_sac,
                           environment = e_trade_gym)

    # df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(model=trained_sac,
    #                        environment = e_trade_gym)



    """### Compare different algorithms"""
    new_start = df_account_value.date.iloc[0]
    new_end = df_account_value.date.iloc[-1]
    df_account_value_td3 = df_account_value_td3[(df_account_value_td3.date >= new_start) & (df_account_value_td3.date < new_end)]
    df_account_value_a2c = df_account_value_a2c[(df_account_value_a2c.date >= new_start) & (df_account_value_a2c.date < new_end)]
    df_account_value_sac = df_account_value_sac[(df_account_value_sac.date >= new_start) & (df_account_value_sac.date < new_end)]
    df_account_value_ensemble = df_account_value[(df_account_value.date >= new_start) & (df_account_value.date < new_end)]

    trade = ts_processor.data_split(ts_processor.dataframe, new_start, new_end)

    """### Backtest"""



    def pct(l):
        """Get percentage"""
        base = l[0]
        return [x / base for x in l]


    def get_baseline(ticket):
      df = tushare.get_hist_data(ticket, start=val_start_date, end=val_stop_date)
      df.loc[:, "dt"] = df.index
      df.index = range(len(df))
      df.sort_values(axis=0, by="dt", ascending=True, inplace=True)
      df["date"] = pd.to_datetime(df["dt"], format="%Y-%m-%d")
      return df


    baseline_label = "Equal-weight portfolio"
    tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}
    baseline_ticket = '399300'
    if baseline_ticket:
        # 使用指定ticket作为baseline
        baseline_df = get_baseline(baseline_ticket)

        baseline_date_list = baseline_df.date.dt.strftime('%Y-%m-%d').tolist()
        df_date_list = df_account_value_ensemble.date.tolist()
        df_account_value_ensemble = df_account_value_ensemble[df_account_value_ensemble.date.isin(baseline_date_list)]
        df_account_value_td3 = df_account_value_td3[df_account_value_td3.date.isin(baseline_date_list)]
        df_account_value_a2c = df_account_value_a2c[df_account_value_a2c.date.isin(baseline_date_list)]
        df_account_value_sac = df_account_value_sac[df_account_value_sac.date.isin(baseline_date_list)]
        baseline_df = baseline_df[baseline_df.date.isin(df_date_list)]

        baseline = baseline_df.close.tolist()
        baseline_label = tic2label.get(baseline_ticket, baseline_ticket)
    else:
        # 均等权重
        all_date = trade.date.unique().tolist()
        baseline = []
        for day in all_date:
            day_close = trade[trade["date"] == day].close.tolist()
            avg_close = sum(day_close) / len(day_close)
            baseline.append(avg_close)

    agent_ensemble = pct(df_account_value_ensemble.account_value.tolist())
    agent_td3 = pct(df_account_value_td3.account_value.tolist())
    agent_a2c = pct(df_account_value_a2c.account_value.tolist())
    agent_sac = pct(df_account_value_sac.account_value.tolist())
    # agent_sac = pct(df_account_value_sac.account_value.tolist())

    baseline = pct(baseline)

    days_per_tick = (
        150  # you should scale this variable accroding to the total trading days
    )
    time = list(range(len(agent_td3)))
    datetimes = df_account_value_td3.date.tolist()

    ticks = [tick for t, tick in zip(time, datetimes) if t % days_per_tick == 0]
    plt.title("Cumulative Returns")
    plt.plot(time, agent_ensemble, label="Ensembled Agent", color="orangered")
    plt.plot(time, agent_td3, label="TD3 Agent", color="gold")
    plt.plot(time, agent_a2c, label="A2C Agent", color="deepskyblue")
    plt.plot(time, agent_sac, label="SAC Agent", color="lightgreen")
    # plt.plot(time, agent_sac, label="SAC Agent", color="red")
    plt.plot(time, baseline, label=baseline_label, color="slategray")
    plt.xticks([i * days_per_tick for i in range(len(ticks))], ticks, fontsize=7)

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid()
    plt.legend()
    plt.axhline(1, color = 'k', linewidth = 3)
    plt.show()
    plt.savefig(f'./plots/csi300_trained_{TIME_STEPS}.png')
    print(f"saved_{TIME_STEPS}")
    # plot_new(trade, baseline_ticket='399300')
    # plot_new(trade)

