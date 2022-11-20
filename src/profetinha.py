import prophet
import pandas as pd
import yfinance as yf

def get_data(stock_symbol): 
    try:
        stock_data = yf.download(stock_symbol, interval = '1d')
        return stock_data
    except:
        return None


def get_future_value(stock_symbol, qtd_dias_futuros):
    stock_data = get_data(stock_symbol)

    df = pd.DataFrame()
    df['y'] = stock_data['Close']
    df['ds'] = stock_data.index

    modelo = prophet.Prophet(daily_seasonality= True)
    modelo.fit(df)

    futuro = modelo.make_future_dataframe(periods= qtd_dias_futuros)
    predictions = modelo.predict(futuro)

    return predictions['yhat'].iloc[-1]