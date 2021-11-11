from neuralprophet import NeuralProphet
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import utils

def lerDados():
    df18 = pd.read_csv('data/lins_2018.CSV', sep=';', header=None)
    df19 = pd.read_csv('data/lins_2019.CSV', sep=';', header=None)
    df20 = pd.read_csv('data/lins_2020.CSV', sep=';', header=None)
    df21 = pd.read_csv('data/lins_2021.CSV', sep=';', header=None)
    result = pd.concat([df18, df19, df20])
    df = utils.preprocess(result)
    df = utils.preencherZero(df)
    df21 = utils.preprocess(df21)
    df21 = utils.preencherZero(df21)
    return [df, df21]


def verDados(df):
    Temp = df['Temperatura']
    Temp.plot(title = 'Temperatura', color='orange')
    plt.ylabel('Temperatura ºC')
    plt.show()


def lerRede():
    rn = pd.read_pickle("data/Prophet.pkl")
    return rn


def treinarRede(df):
    rn = NeuralProphet(growth="discontinuous",
                    changepoints=None,
                    n_changepoints=5,
                    changepoints_range=0.8,
                    trend_reg = 0,
                    trend_reg_threshold=False,
                    yearly_seasonality='auto',
                    weekly_seasonality='auto',
                    daily_seasonality='auto',
                    seasonality_mode=True,
                    seasonality_reg=0,
                    n_forecasts=1,
                    n_lags=0,
                    num_hidden_layers=100,
                    d_hidden=50,
                    ar_sparsity=None,
                    learning_rate=None,
                    epochs=1000,
                    loss_func='Huber',
                    normalize='auto',
                    impute_missing=True,
                    )
    rn.fit(df, freq='H')
    return rn


def testarRede(rn, df,df21):
    dia = 24
    semana = 24*7
    mes = dia*30
    bimestre = mes*2
    ano = mes*12
    periodo = 20*dia    #20 dias
    future = rn.make_future_dataframe(df, periods=ano)
    forecast = rn.predict(future)
    forecast = forecast[['ds','yhat1']].set_index('ds').squeeze()
    df21[0:periodo].plot(label='real', color='g')
    forecast[0:periodo].plot(label='Previsão', color='b')
    plt.legend()
    plt.show()


def salvarRede(rn):
    pkl_path = "Prophet.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(rn, f)


def main():
    op = 0  # 0 - ler rede;  1 - treinar rede
    df, df21 = lerDados()
    verDados(df)
    df.reset_index(inplace = True)
    df.columns = ['ds','y']
    if op:
        rn = treinarRede(df)
    else:
        rn = lerRede()
    testarRede(rn, df, df21)
    salvarRede(rn)

if __name__ == '__main__':
    main()