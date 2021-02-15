from __future__ import division, print_function
from flask import jsonify, make_response,flash
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import time
import json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import datetime
from dateutil import relativedelta
from xgboost import XGBRegressor
import pandas as pd
from flask import send_file

app = Flask(__name__)






def create_xtest(d):
    #how about it when the user asks for old results not preds:  ask for future date
    period=int(d["period"])
    shop_id=d["shop_id"]=int(d["shop_id"])
    item_id=d["item_id"]=int(d["item_id"])
    try:
        item_category=d["item_category"]=int(d["item_category"])
    except:
        flash("no strings allowed !")
        return render_template('index.html')
    id_struct=d["id_struct"]=int(d["id_struct"])
    price=d["price"]=float(d["price"])
    date=d["date"]=d["date"]
    dd={}
    lss=date.split("-")
    dates=[datetime.date(int(lss[0]),int(lss[1]), 1)]
    for i in range(period-1):
        dates.append(dates[-1] + relativedelta.relativedelta(months=1))
    dd["Date"]=dates
    dd["shop_id"]=[shop_id]*period
    dd["item_id"]=[item_id]*period
    dd["item_category"]=[item_category]*period
    dd["id_struct"]=[id_struct]*period
    dd["Price_agg"]=[price]*period
    s = pd.DataFrame(data=dd)
    s["Date"]=pd.to_datetime(s["Date"])
    try:
        holy=pd.read_csv("./data/base_data/holidaysprime.csv")
    except:
        holy=pd.read_csv("./data/base_data/holidays.csv")
    holy["date"] = holy["year"].astype(str).str.cat(holy[["month", "day_of_month"]].astype(str), sep="-")
    holy["Date"]=pd.to_datetime(holy["date"])
    holy=holy.drop(["date","year","month","day_of_month"],axis=1)
    holy["key"]=1
    dff=holy.set_index(['Date'])
    dff1=dff.groupby(pd.Grouper(freq="M")).sum()
    dff2=dff1.fillna(0).reset_index()
    ll1=[x.replace(day=1) for x in dff2.Date]
    dff2["Date"]=ll1
    dff3=dff2.set_index(['Date'])
    kezz=[]
    for i in s.Date:
        try:
            kj=int(dff3.loc[i])
            kezz.append(kj)
        except:
            kj=0
            kezz.append(kj)
    s["keyz"]=kezz
    #loading base dataset
    try:
        base=pd.read_csv("./data/base_data/base_dataprime.csv")
    except:
        base=pd.read_csv("./data/base_data/base_data.csv")
    base["Date"]=pd.to_datetime(base["Date"])
    last_block_num=int(list(base.date_block_num.values)[-1])
    last_date=list(base.Date)[-1]
    first_new_date=list(s.Date)[0]
    if first_new_date>=last_date:
        num_months = (first_new_date.year - last_date.year) * 12 + (first_new_date.month - last_date.month)
        new_block_num=num_months+last_block_num
    else:
        num_months = (last_date.year - first_new_date.year) * 12 + (last_date.month - first_new_date.month)
        new_block_num=last_block_num-num_months
    ks=[new_block_num]
    for i in range(len(s.Date)-1):
        ks.append(ks[-1]+1)
    s["date_block_num"]=ks
    s["item_cnt_month_lag1"],s["item_cnt_month_lag2"],s["item_cnt_month_lag3"],s["item_cnt_month_lag4"],s["item_cnt_month_lag5"],s["item_cnt_month_lag6"],s["item_cnt_month_lag7"],s["Price_agg_lag1"],s["Price_agg_lag2"]=0,0,0,0,0,0,0,0,0
    ds=base[["Date","date_block_num","shop_id","item_id","item_category","id_struct","Price_agg","item_cnt_month"]]
    try:
        grouped_XT = ds.groupby(["shop_id", "item_id"]).get_group((shop_id,item_id))
    except:
        #ouput console messege
        print("no pair in dataset !!!")
    XT_list=[0]*10
    for i in range(10):
        try:
            XT_list[i]=list(grouped_XT.item_cnt_month.values)[-(i+1)]
        except:
            break
    Xs_list=[0]*5
    for i in range(5):
        try:
            Xs_list[i]=list(grouped_XT.Price_agg.values)[-(i+1)]
        except:
            break
    for i in range(len(s.Date)):
        s.iloc[i,8:-2]=XT_list[:7]
        XT_list=XT_list[1:]
    for i in range(len(s.Date)):
        s.iloc[i,-2:]=Xs_list[:2]
        Xs_list=Xs_list[1:]
    return s




def create_trainable_dataset(df):
    #loading base dataset
    try:
        base=pd.read_csv("./data/base_data/base_dataprime.csv")
    except:
        base=pd.read_csv("./data/base_data/base_data.csv")
    base["Date"]=pd.to_datetime(base["Date"])
    last_block_num=int(list(base.date_block_num.values)[-1])
    last_date=list(base.Date)[-1]
    df["Date"]=pd.to_datetime(df["Date"])
    first_new_date=list(df.Date)[0]
    if first_new_date>=last_date:
        num_months = (first_new_date.year - last_date.year) * 12 + (first_new_date.month - last_date.month)
        new_block_num=num_months+last_block_num
    else:
        num_months = (last_date.year - first_new_date.year) * 12 + (last_date.month - first_new_date.month)
        new_block_num=last_block_num-num_months

    l=[]
    old=1
    #change count
    count=new_block_num
    for i in df.Date:
        if float(i.month)!=float(old):
            old=float(i.month)
            count=count+1
        l.append(count)
    df["date_block_num"]=l

    daata=df

    #add empty dates
    start=daata.sort_values(by="Date").Date.values[0]
    end=daata.sort_values(by="Date").Date.values[-1]
    l=str(start).split('-')
    ll=str(end).split('-')
    start=l[0]+"-"+l[1]+"-"+l[2][:2]
    end=ll[0]+"-"+ll[1]+"-"+ll[2][:2]
    d={'Date':pd.date_range(start=start,end=end)}
    temp_df = pd.DataFrame(data=d)

    temp=daata[["shop_id","item_id","item_category","id_struct"]].drop_duplicates(["shop_id","item_id"]).reset_index()

    temp=temp.drop(["index"],axis=1)

    temp1=daata.drop_duplicates(["item_id"],keep='last').reset_index()
    temp2=temp1[["item_id","Price"]].set_index(["item_id"])

    temp['key'] = 0
    temp_df['key'] = 0

    df_cartesian = temp_df.merge(temp, how='outer')

    combination=df_cartesian.drop(["key"],axis=1)

    temppp=daata.drop(["date_block_num","item_category","id_struct"],axis=1)

    final_comb=pd.merge(combination,temppp , on=["Date","shop_id","item_id"], how="left")

    lis=list(final_comb.Price.fillna(-1).values)
    lis1=list(final_comb.item_id.values)

    kk=[lis[x] if lis[x]!=-1 else float(temp2.loc[lis1[x]]) for x in range(len(lis))]

    final=final_comb.drop(["Price"],axis=1)

    final["Price"]=kk
    final=final.fillna(0)

    temp_df.reset_index(inplace=True)
    temp_df=temp_df.drop(["key"],axis=1)

    finals=pd.merge(final,temp_df , on=["Date"], how="left")

    #generating new features
    #holy
    try:
        holy=pd.read_csv("./data/base_data/holidaysprime.csv")
    except:
        holy=pd.read_csv("./data/base_data/holidays.csv")
    holy["date"] = holy["year"].astype(str).str.cat(holy[["month", "day_of_month"]].astype(str), sep="-")
    holy["Date"]=pd.to_datetime(holy["date"])
    holy=holy.drop(["date","year","month","day_of_month"],axis=1)
    holy["key"]=1
    finals1=pd.merge(finals,holy , on=["Date"], how="left")
    finals1=finals1.fillna(0)

    finals1["holidays"] = finals1["holiday"].astype('category')
    finals1["holidayz"] = finals1["holidays"].cat.codes
    finals2=finals1.drop(["holidays","holidayz","holiday"],axis=1)

    #monthly_Agg
    l=[]
    old=1
    #change count
    count=new_block_num
    for i in finals2.Date:
        if float(i.month)!=float(old):
            old=float(i.month)
            count=count+1
        l.append(count)
    finals2["date_block_num"]=l
    finals2["item_cnt_month"] = finals2.groupby(["date_block_num", "shop_id", "item_id"])["item_cnt_day"].transform(np.sum)
    finals2["Price_agg"] = finals2.groupby(["date_block_num", "shop_id", "item_id"])["Price"].transform(np.mean)
    finals2["keyz"] = finals2.groupby(["date_block_num", "shop_id", "item_id"])["key"].transform(np.sum)
    del finals2["item_cnt_day"]
    del finals2["Price"]
    del finals2["index"]
    del finals2["key"]

    finals2 = finals2.drop_duplicates(["date_block_num", "shop_id", "item_id"])

    finals2.reset_index(inplace=True)
    finals2.drop(["index"],axis=1,inplace=True)

    #feature engineering lag roll expand
    grouped_df = finals2.groupby(["shop_id", "item_id"])
    df_list=[]
    for key,_ in grouped_df:
        df_g = grouped_df.get_group(key)
        #lagg features for price and count
        #count
        df_g["item_cnt_month_lag1"]=df_g.item_cnt_month.shift(1).fillna(0)
        df_g["item_cnt_month_lag2"]=df_g.item_cnt_month.shift(2).fillna(0)
        df_g["item_cnt_month_lag3"]=df_g.item_cnt_month.shift(3).fillna(0)
        df_g["item_cnt_month_lag4"]=df_g.item_cnt_month.shift(4).fillna(0)
        df_g["item_cnt_month_lag5"]=df_g.item_cnt_month.shift(5).fillna(0)
        df_g["item_cnt_month_lag6"]=df_g.item_cnt_month.shift(6).fillna(0)
        df_g["item_cnt_month_lag7"]=df_g.item_cnt_month.shift(7).fillna(0)
        #price
        df_g["Price_agg_lag1"]=df_g.Price_agg.shift(1).fillna(0)
        df_g["Price_agg_lag2"]=df_g.Price_agg.shift(2).fillna(0)
        df_list.append(df_g)
    new_df=pd.concat(df_list,axis=0)

    new_df.reset_index(inplace=True)
    new_df.drop(["index"],axis=1,inplace=True)
    return new_df








def model_predict(s):

    param={'colsample_bytree': 0.8, 'subsample': 0.75, 'eta': 0.02, 'n_estimators': 1100, 'max_depth': 7, 'min_child_weight': 1}
    model = XGBRegressor(**param)
    try:
        model.load_model("./models/xgbmodelprime")
    except:
        model.load_model("./models/xgbmodel")
    y_pred = model.predict(s[["date_block_num","shop_id","item_id","id_struct","item_category","Price_agg","keyz","item_cnt_month_lag1","item_cnt_month_lag2","item_cnt_month_lag3","item_cnt_month_lag4","item_cnt_month_lag5","item_cnt_month_lag6","item_cnt_month_lag7","Price_agg_lag1","Price_agg_lag2"]])
    #create current preds file 
    s["predictions"]=y_pred
    c_pred=s[["Date","shop_id","item_id","predictions"]]
    c_pred["Date"]=c_pred["Date"].astype("str")
    try:
        h_preds=pd.read_csv("data/prediction/h_predictions.csv")
    except:
        hp_df = pd.DataFrame({'Date': pd.Series([], dtype='str'),
                    'shop_id': pd.Series([], dtype='int'),
                    'item_id': pd.Series([], dtype='int'),
                   'predictions': pd.Series([], dtype='float')})
        hp_df["Date"]=pd.to_datetime(hp_df["Date"])
        hp_df.to_csv("data/prediction/h_predictions.csv",index=False)
        h_preds=pd.read_csv("data/prediction/h_predictions.csv")
    new_dff=pd.concat([h_preds[["Date","shop_id","item_id","predictions"]],c_pred],axis=0,sort=False)
    new_dff1=new_dff.drop_duplicates(["Date","shop_id","item_id"]).reset_index().drop(["index"],axis=1)
    new_dff1.to_csv("data/prediction/h_predictions.csv",index=False)    
    return s[["Date","shop_id","item_id","predictions"]]


def train_trainable(data):
    df0=data[["date_block_num","shop_id","item_id","id_struct","item_category","Price_agg","keyz","item_cnt_month_lag1","item_cnt_month_lag2","item_cnt_month_lag3","item_cnt_month_lag4","item_cnt_month_lag5","item_cnt_month_lag6","item_cnt_month_lag7","Price_agg_lag1","Price_agg_lag2"]]
    df1=data[["item_cnt_month"]]  
    param={'colsample_bytree': 0.8, 'subsample': 0.75, 'eta': 0.02, 'n_estimators': 1100, 'max_depth': 7, 'min_child_weight': 1}
    model = XGBRegressor(**param)
    model.fit(
            df0, 
            df1, 
            eval_metric="rmse", 
            eval_set=[(df0, df1)], 
            verbose=False, 
            early_stopping_rounds = 1)
    model.save_model("./models/xgbmodelprime")


@app.route('/forecast/', methods=['GET','POST'])
def process():
    if request.method == 'POST':
        #here file holidays or data
        if request.files :
            filee=request.files["myfile"]
            its_name=filee.filename
            #delete it
            for filename in os.listdir('./uploads/'): 
                    os.remove('./uploads/' + filename)    
            #save_it
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(its_name))
            filee.save(file_path)      
            if its_name=="holidays.csv":
                try:
                    df=pd.read_csv(file_path)
                    if list(df.columns)!=["year","month","day_of_month","holiday"]:
                        flash("corrupt file or format!")
                        return render_template('index.html')
                except:
                    flash("corrupt file or format!")
                    return render_template('index.html')
                try:
                    holy=pd.read_csv("./data/base_data/holidaysprime.csv")
                except:
                    holy=pd.read_csv("./data/base_data/holidays.csv")
                res=pd.concat([holy,df])
                res=res.drop_duplicates(["year","month","day_of_month"],keep='last')
                res=res.reset_index(drop=True)
                res.to_csv("./data/base_data/holidaysprime.csv",index=False)
                flash("holidays added !")
            else:
                try:
                    df=pd.read_excel(file_path,engine='openpyxl')
                    if list(df.columns)!=['Date', 'shop_id', 'item_id', 'item_category', 'id_struct', 'Price', 'item_cnt_day']:
                        flash("corrupt file or format!")
                        return render_template('index.html')
                except:
                    flash("corrupt file or format!")
                    return render_template('index.html')
                #call create trainable dataset
                ddf=create_trainable_dataset(df)
                #fuse ddf and base
                #loading base dataset
                try:
                    base=pd.read_csv("./data/base_data/base_dataprime.csv")
                except:
                    base=pd.read_csv("./data/base_data/base_data.csv")
                base["Date"]=pd.to_datetime(base["Date"])
                resultt=pd.concat([base,ddf])
                resultt=resultt.drop_duplicates(["Date","shop_id","item_id"],keep='last')
                resultt=resultt.reset_index(drop=True)
                resultt.to_csv("./data/base_data/base_dataprime.csv",index=False)
                train_trainable(resultt)
                flash("historical data added and model updated !")
            return render_template('index.html')
        else:
            d=dict(request.form)
            try:
                shop_id=d["shop_id"]=int(d["shop_id"])
                item_id=d["item_id"]=int(d["item_id"])
            except:
                flash("no strings allowed !")
                return render_template('index.html')
            #extract ground truth
            try:
                #loading base dataset
                try:
                    base=pd.read_csv("./data/base_data/base_dataprime.csv")
                except:
                    base=pd.read_csv("./data/base_data/base_data.csv")
                base["Date"]=pd.to_datetime(base["Date"])
                grouped_gt = base.groupby(["shop_id", "item_id"]).get_group((shop_id,item_id))
            except:
                #ouput console messege
                flash("This (store,item) pair dosen't exist in  the database !")
                return render_template('index.html')
            gtt=grouped_gt[["Date","item_cnt_month"]]
            #extract historical preds
            try:
                h_preds=pd.read_csv("./data/prediction/h_predictions.csv")
                #if issue output console message
                grouped_hp = h_preds.groupby(["shop_id", "item_id"]).get_group((shop_id,item_id))
                grouped_hp["Date"]=pd.to_datetime(grouped_hp["Date"])
                hp=grouped_hp[["Date","predictions"]]
            except:
                hp=pd.DataFrame(columns = ['Date','predictions']) 
            try:
                x_test=create_xtest(d)
                c_preds=model_predict(x_test)
            except:
                flash("no strings allowed !")
                return render_template('index.html')
            return render_template('index.html',data=gtt.to_json(),data1=hp.to_json(),data2=c_preds.to_json())
    elif request.method == 'GET':
        if os.path.isfile("./data/base_data/base_dataprime.csv"):
            os.remove("./data/base_data/base_dataprime.csv") 
        if os.path.isfile("./data/base_data/holidaysprime.csv"):
            os.remove("./data/base_data/holidaysprime.csv") 
        if os.path.isfile("./models/xgbmodelprime"):
            os.remove("./models/xgbmodelprime") 
        if os.path.isfile("./data/prediction/h_predictions.csv"):
            os.remove("./data/prediction/h_predictions.csv") 
        flash("application reset !")
        return render_template('index.html')
    
 


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

    

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
