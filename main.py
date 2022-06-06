import json
import time
import requests
import datetime
import base64
import mplfinance as mpf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from pytrends.request import TrendReq
from pytrends import dailydata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from flask import Flask, render_template, flash, redirect, url_for, request
from io import BytesIO


app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "secretkey" # secret_key


kayitli_modeller = {
    "BTCUSD" : "./static/kayitlimodeller/BTCUSD.h5",
    "ETHUSD" : "./static/kayitlimodeller/ETHUSD.h5"
}

def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=False, input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear"))
    model.compile(loss="mse", optimizer="adam")
    return model

def model_veri(df):
    X_train, X_test, y_train, y_test = [], [], [], []
    scaler = MinMaxScaler()
    scaler.fit(df[["close"]].values)
    dataset = df.drop(["volume"], axis=1)
    scaler2 = MinMaxScaler()
    dataset_sc = scaler2.fit_transform(dataset)
    train_data, test_data = train_test_split(dataset_sc, test_size=0.2, shuffle=False)
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i,:])
        y_train.append(train_data[i,0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    for i in range(60,len(test_data)):
        X_test.append(test_data[i-60:i,:])
        y_test.append(test_data[i,0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))

    return X_train, X_test, y_train, y_test



def veri_al(kripto="ETH", birim="USD", aralik="day", aggregate=1, periyot="max"):
    ex = birim
    aralik = aralik if aralik in ["day","hour","minute"] else "day" # day - hour - minute
    aggregate = aggregate # (saatlik ve dakikalık için 0-30)
    limitdata = 2000 if aralik=="day" else 2000//aggregate
    toTs= ""
    sorgu = f"&aggregate={aggregate}" if aralik != "day" else ""
    api_key = '' # cryptocompare api_key
    api_url = f'https://min-api.cryptocompare.com/data/v2/histo{aralik}?limit={limitdata}{sorgu}&fsym={kripto}&tsym={ex}&api_key={api_key}'
    api_url_vol = f'https://min-api.cryptocompare.com/data/exchange/histo{aralik}?&limit={limitdata}{sorgu}&tsym={kripto}&api_key={api_key}'
    
    res = requests.get(api_url).json()
    
    df = pd.DataFrame(res['Data']['Data'])[['time', 'high', 'low', 'open', 'close']].set_index('time')
    df.index = pd.to_datetime(df.index, unit = 's')
    if aralik == "minute":
        df["volume"] = np.nan
    else:
        res2 = requests.get(api_url_vol).json()
        df2 = pd.DataFrame(res2["Data"]).set_index("time")
        df2.index = pd.to_datetime(df2.index, unit = 's')
        df = pd.concat([df,df2], axis=1)

    if aralik in ["hour","minute"]:
        for i in range(4 if aralik=="minute" else 7):
            timestamp = int(datetime.datetime.timestamp(df.index[0]))
            toTs= f"&toTs={timestamp}"
            api_url_toTs = f'https://min-api.cryptocompare.com/data/v2/histo{aralik}?limit={limitdata}{sorgu}&fsym={kripto}&tsym={ex}{toTs}&api_key={api_key}'
            api_url_vol_toTs = f'https://min-api.cryptocompare.com/data/exchange/histo{aralik}?&limit={limitdata}{sorgu}&tsym={kripto}{toTs}&api_key={api_key}'
            rests = requests.get(api_url_toTs).json()
            dfTs = pd.DataFrame(rests['Data']['Data'])[['time', 'high', 'low', 'open', 'close']].set_index('time')
            dfTs.index = pd.to_datetime(dfTs.index, unit = 's')
            if aralik == "minute":
                dfTs["volume"] = np.nan
            else:
                restsvol = requests.get(api_url_vol_toTs).json()
                dfTsVol = pd.DataFrame(restsvol["Data"]).set_index("time")
                dfTsVol.index = pd.to_datetime(dfTsVol.index, unit = 's')
                dfTs = pd.concat([dfTs,dfTsVol], axis=1)
            df = pd.concat([dfTs, df])
    today = datetime.datetime.today().date()
    if periyot=="max":
        return df
    elif periyot.endswith("d"):
        agg = periyot[:-1]
        return df[today-pd.offsets.Day(int(agg)):]
    elif periyot.endswith("mo"):
        agg = periyot[:-2]
        return df[today-pd.DateOffset(months=int(agg)):]
    elif periyot.endswith("y"):
        agg = periyot[:-1]
        return df[today-pd.DateOffset(years=int(agg)):]
    else:
        return df


def grafik(kripto, birim, aralik, periyot):
    if aralik.endswith("d"):
        veriarlk = "day"
    elif aralik.endswith("h"):
        veriarlk = "hour"
    elif aralik.endswith("m"):
        veriarlk = "minute"
    agg = int(aralik[:-1])
    df = veri_al(kripto,birim,veriarlk,agg, periyot)
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def sinyal(df):
    df["SMA30"] = df["close"].rolling(30).mean()
    df["SMA100"] = df["close"].rolling(100).mean()
    tp = (df["close"] + df["low"] + df["high"])/3
    tpstd = tp.rolling(20).std(ddof=0)
    ma_tp = tp.rolling(20).mean()
    df["BOLU"] = ma_tp + 2*tpstd
    df["BOLD"] = ma_tp - 2*tpstd
    exp1 = df['close'].ewm(span=12, adjust=False, min_periods=12).mean()
    exp2 = df['close'].ewm(span=26, adjust=False, min_periods=26).mean()
    macd = exp1 - exp2
    macdexp = macd.ewm(span=9, adjust=False).mean()

    masignalbuy, masignalsell = [], []
    position = False
    for i in range(len(df)):
        if df['SMA30'][i] > df['SMA100'][i]:
            if position == False :
                masignalbuy.append(df['close'][i])
                masignalsell.append(np.nan)
                position = True
            else:
                masignalbuy.append(np.nan)
                masignalsell.append(np.nan)
        elif df['SMA30'][i] < df['SMA100'][i]:
            if position == True:
                masignalbuy.append(np.nan)
                masignalsell.append(df['close'][i])
                position = False
            else:
                masignalbuy.append(np.nan)
                masignalsell.append(np.nan)
        else:
            masignalbuy.append(np.nan)
            masignalsell.append(np.nan)
    df["SMA_BUY"] = masignalbuy
    df["SMA_SELL"] = masignalsell

    MACD_Buy=[]
    MACD_Sell=[]
    position=False

    for i in range(0, len(df)):
        if macd[i] > macdexp[i] :
            MACD_Sell.append(np.nan)
            if position ==False:
                MACD_Buy.append(df['close'][i])
                position=True
            else:
                MACD_Buy.append(np.nan)
        elif macd[i] < macdexp[i] :
            MACD_Buy.append(np.nan)
            if position == True:
                MACD_Sell.append(df['close'][i])
                position=False
            else:
                MACD_Sell.append(np.nan)
        elif position == True and df['close'][i] < MACD_Buy[-1] * (1 - 0.025):
            MACD_Sell.append(df["close"][i])
            MACD_Buy.append(np.nan)
            position = False
        elif position == True and df['close'][i] < df['close'][i - 1] * (1 - 0.025):
            MACD_Sell.append(df["close"][i])
            MACD_Buy.append(np.nan)
            position = False
        else:
            MACD_Buy.append(np.nan)
            MACD_Sell.append(np.nan)

    df["MACD_BUY"] = MACD_Buy
    df["MACD_SELL"] = MACD_Sell
        

    bbBuy = []
    bbSell = []
    position = False

    for i in range(len(df)):
        if df['close'][i] < df['BOLD'][i]:
            if position == False :
                bbBuy.append(df['close'][i])
                bbSell.append(np.nan)
                position = True
            else:
                bbBuy.append(np.nan)
                bbSell.append(np.nan)
        elif df['close'][i] > df['BOLU'][i]:
            if position == True:
                bbBuy.append(np.nan)
                bbSell.append(df['close'][i])
                position = False
            else:
                bbBuy.append(np.nan)
                bbSell.append(np.nan)
        else :
            bbBuy.append(np.nan)
            bbSell.append(np.nan)

    df["BOLL_BUY"] = bbBuy
    df["BOLL_SELL"] = bbSell

    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tahmin")
def tahminsayfa():
    return render_template("tahmin.html")

@app.route("/model")
def tahmin():
    if "smb" in request.args and "birim" in request.args:
        kripto = request.args.get("smb")
        birim = request.args.get("birim")
        aralik = request.args.get("aralik", default="1d")
        if aralik == "1d":
            veriarlk = "day"
            agg = 1
        elif aralik == "4h":
            veriarlk = "hour"
            agg = 4
        elif aralik == "15m":
            veriarlk = "minute"
            agg = 15
        df = veri_al(kripto=kripto, birim=birim,aralik=veriarlk, aggregate=agg)
        X_train, X_test, y_train, y_test = model_veri(df)
        if kripto+birim in kayitli_modeller:
            model = create_model(X_train)
            model.load_weights(kayitli_modeller[kripto+birim])
        else:
            model = create_model(X_train)
            model.fit(X_train, y_train, epochs=25, batch_size=32)
        scaler = MinMaxScaler()
        scaler.fit(df[["close"]].values)
        scaler2 = MinMaxScaler()
        df_sc = scaler2.fit_transform(df.drop(["volume"],axis=1))
        son60 = df_sc[-60:]
        pred = model.predict(son60.reshape(1,son60.shape[0],son60.shape[1]))
        sinyallist = []
        df = sinyal(df)
        if df["BOLL_BUY"].dropna().iloc[[-1]].index > df["BOLL_SELL"].dropna().iloc[[-1]].index:
            sinyallist.append("AL")
        else:
            sinyallist.append("SAT")
        if df["SMA_BUY"].dropna().iloc[[-1]].index > df["SMA_SELL"].dropna().iloc[[-1]].index:
            sinyallist.append("AL")
        else:
            sinyallist.append("SAT")
        if df["MACD_BUY"].dropna().iloc[[-1]].index > df["MACD_SELL"].dropna().iloc[[-1]].index:
            sinyallist.append("AL")
        else:
            sinyallist.append("SAT")
        fig, ax1 = plt.subplots()
        ax1.plot(df.index, df.close, zorder=1, label=kripto+"-"+birim)
        ax1.scatter(df.index, df["SMA_BUY"], color='green', marker='^', s=80, zorder=2)
        ax1.scatter(df.index, df["SMA_SELL"], color='red', marker='v', s=80, zorder=2)
        ax1.legend(loc="upper left")
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())
        figfile.close()
        print(sinyallist)
        return json.dumps({
            "tarih" : str(df.index[-1].strftime("%d.%m.%Y")),
            "kripto" : kripto+"-"+birim,
            "sonkapanis" : str(df.close.values[-1]),
            "tahmin" : str(round(float(scaler.inverse_transform(pred.reshape(-1,1))),4)),
            "sinyal" : max(sinyallist, key=sinyallist.count),
            "grafik" : figdata_png.decode("utf-8")
        })
    else:
        return "Bad Request!", 400


@app.route("/verial")
def veri():
    if len(request.args) == 4:
        kripto = request.args.get("smb")
        birim = request.args.get("birim")
        aralik = request.args.get("aralik")
        periyot = request.args.get("per")
        return grafik(kripto,birim,aralik,periyot)
    else:
        return "Bad Request!", 400

@app.route("/grafik")
def grafikpng():
    if request.method == "GET" and "tur" in request.args:
        kripto = request.args.get("smb")
        birim = request.args.get("birim")
        aralik = request.args.get("aralik")
        periyot = request.args.get("per")
        if aralik.endswith("d"):
            veriarlk = "day"
        elif aralik.endswith("h"):
            veriarlk = "hour"
        elif aralik.endswith("m"):
            veriarlk = "minute"
        agg = int(aralik[:-1])
        if request.args.get("tur") == "mplfinance":
            df = veri_al(kripto,birim,veriarlk,agg, periyot)
            figfile = BytesIO()
            mpf.plot(df,
            type='candle',
            style='charles',
            ylabel='Fiyat',
            volume=False if df["volume"].isna().any() else True,
            ylabel_lower='Hacim',savefig=figfile)
            plt.savefig(figfile, format='png')
            figfile.seek(0)
            figdata_png = base64.b64encode(figfile.getvalue())
            figfile.close()
            return {"lastopen": df["open"][-1], "lastclose": df["close"][-1], "grafik":figdata_png.decode("utf-8")}
        elif request.args.get("tur") == "cizgigr":
            df = veri_al(kripto,birim,veriarlk,agg, periyot)
            delta = df.close.diff()
            up = delta.clip(lower=0)
            down = -1*delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up/ema_down
            df["RSI"] = 100 - (100/(1+rs))
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
            ax1.plot(df.index,df.close)
            ax2.plot(df.index,df.RSI, color="yellow")
            ax1.set_ylabel(kripto+" Fiyat")
            ax2.set_ylabel("RSI")
            figfile = BytesIO()
            plt.savefig(figfile, format='png')
            figfile.seek(0)
            figdata_png = base64.b64encode(figfile.getvalue())
            figfile.close()
            return {"lastopen": df["open"][-1], "lastclose": df["close"][-1], "grafik":figdata_png.decode("utf-8")}
        else:
            return ""
    else:
        return "Bad Request!", 400

if __name__ == "__main__":
    app.run()
