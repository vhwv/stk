
# -*- coding:utf-8 -*- 

import pandas as pd
import numpy as np
import tushare as ts
import datetime
import os,sys
sys.path.append("/Users/wyatt/Documents/Docs-wMBP15/SAS/stk_model/python/lib")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import FontManager, FontProperties 

def chnchr():  
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')  

pth = "/Users/wyatt/Documents/Docs-wMBP15/Stocks/Data/"


def stk_daily_today():
    '''
    Download today's ohlc for all stocks and savd to daily raw file 
    '''
    dtf=datetime.date.today().strftime('%Y%m%d')
    print('Downloading data on : %s' %(dtf))
    df = ts.get_today_all()
    print('\nDownload data completed.\nNow will continue to process data:')

    df.rename(inplace=True,columns={'per':'pe','mktcap':'mv','nmc':'liqmv','changepercent':'pctchange','settlement':'preclose','trade':'close','turnoverratio':'turn'})
    df['sid']=df.code.str[:]
    df['month'] = int(dtf)
    df['change']=df.close-df.preclose
    df['amplitude'] = (df.high-df.low)/df.preclose
    #df['highlimit'] =list(map(lambda x: 1 if x == min(tt_tmp['price']) else(0), tt_tmp['price']))
    df['highlimit']=list(map(lambda x,y: '是' if x >=round(y*1.1,2) else('否'), df.close,df.preclose))
    df['lowlimit']=list(map(lambda x,y: '是' if x <=round(y*0.9,2) else('否'), df.close,df.preclose))
    df['code']=list(map(lambda x: x+'.SH' if x[:1]==str(6) else(x+'.SZ'), df.code))
    df = df[df.volume>0]
    df=df[(df.code.str[:1] == '0')|(df.code.str[:1] == '3')|(df.code.str[:1] == '6')]
    '''
    for idx,r in df.iterrows():
        if r.code[:1] == str(6):
            df.loc[idx,'code']=df.loc[idx,'code'] + '.SH'
        else:
            df.loc[idx,'code']=df.loc[idx,'code'] + '.SZ'
    '''
    df=df.sort_values('code').reset_index(drop=True)

    fp = "/Users/wyatt/Documents/Docs-wMBP15/Stocks/Data/"
    fn='daily_raw_%s.xlsx' %(dtf)
    ffn=fp + fn
    df.to_excel(ffn,index=False)
    print("Download data saved as file: %s" %(ffn))
    return df

def chk_dt(dt):
    dt_tmp=str(dt)
    dt_tmp = str(dt_tmp).replace('-','')
    dt_tmp = str(dt_tmp).replace('/','')
    dt_tmp = str(dt_tmp).replace('\\','')
    dt_tmp = str(dt_tmp).replace('.','')
    if len(dt_tmp)>8 or len(dt_tmp)<6:
        print("%s : Date length not recognized. Please use valid Date format YYYYMMDD or YYMMDD." %(dt_tmp)) 
        return None
    try:
        if len(dt_tmp)==6:
            dtt = datetime.datetime.strptime(str(dt_tmp),"%y%m%d").date()
        else:
            dtt = datetime.datetime.strptime(str(dt_tmp),"%Y%m%d").date()
    except:
        print("%s : Date format not recognized. Please use valid Date format YYYYMMDD or YYMMDD." %(dt_tmp))    
        return None
    return dtt
                 

def stk_daily_analysis(dsn):
    tt=dsn.copy()
    tt = tt[np.logical_not(tt.name.str.contains('ST'))]
    tt=tt.sort_values('pctchange',ascending=False).reset_index(drop=True)

    tt['pct_oc']=tt.open/tt.preclose - 1
    tt['pct_lc']=tt.low/tt.preclose - 1
    tt['pct_hc']=tt.high/tt.preclose-1
    tt['pct_cc']=tt.close/tt.preclose-1
    tt['oeql'] = tt.open==tt.low
    tt['ceqh'] = tt.close==tt.high

    bins=list(np.array(range(-110,110,5))/1000)
    tt['bin_oc'] = list(pd.cut(tt.pct_oc,bins))
    tt['bin_lc'] = list(pd.cut(tt.pct_lc,bins))
    tt['bin_hc'] = list(pd.cut(tt.pct_hc,bins))
    tt['bin_cc'] = list(pd.cut(tt.pct_cc,bins))
    return tt

def stk_daily_hist(_sdt,_edt=None,save=True):
    if _edt == None:
        _edt=_sdt
        
    if chk_dt(_sdt) ==None or chk_dt(_edt)==None:
        return None
    
    sdt_dt = chk_dt(_sdt)
    sdt_df = sdt_dt.strftime("%Y-%m-%d")
    sdt_ds = sdt_dt.strftime("%Y%m%d") 
    
    edt_dt = chk_dt(_edt)
    edt_df = edt_dt.strftime("%Y-%m-%d")
    edt_ds = edt_dt.strftime("%Y%m%d")
    
    dcnt = (edt_dt-sdt_dt).days

    df_all=pd.DataFrame()
    dt = sdt_dt
    df=ts.get_day_all(sdt_df)
    while dt <= edt_dt:        
        dt_df=dt.strftime("%Y-%m-%d")
        dt_ds=dt.strftime("%Y%m%d")
        try:
            df =ts.get_day_all(dt_df)           
        except:
            dt = dt + datetime.timedelta(days=1)
            continue
        print('Data download completed on : %s' %(dt_ds))
        df['month']= int(dt_ds)
        df_all = df_all.append(df)
        dt = dt + datetime.timedelta(days=1)
    df_all['sid']=df_all.code
    df_all['code']=list(map(lambda x: x+'.SH' if x[0]==6 else(x+'.SZ'),df_all.sid))
    df_all=df_all[df_all.volume>0]
    df_all.rename(columns={'turnover':'turn','range':'amplitude','selling':'buyvol','buying':'sellvol','totals':'totalshare','floats':'liqshare','fvalues':'mva2','abvalues':'mv','avgprice':'average','preprice':'preclose','p_change':'pctchange','price':'close'},inplace=True)
    df_all['highlimit']=list(map(lambda x,y: '是' if x >=round(y*1.1,2) else('否'), df_all.close,df_all.preclose))
    df_all['lowlimit']=list(map(lambda x,y: '是' if x <=round(y*0.9,2) else('否'), df_all.close,df_all.preclose))
    df_all.amplitude = df_all.amplitude/100
    df_all.volume=df_all.volume*100
    df_all.buyvol=df_all.buyvol*100
    df_all.sellvol=df_all.sellvol*100    
    df_all.totalshare=df_all.totalshare*10000
    df_all.liqshare=df_all.liqshare*10000
    df_all.mva2=df_all.mva2*100000000
    df_all.mv=df_all.mv*100000000
    df_all.sort_values(['month','sid'],inplace=True)
    df_all.reset_index(drop=True,inplace=True)
    print("All data download completed.")

    if save==True:
        print("Continue to Process...")
        fp = "/Users/wyatt/Documents/Docs-wMBP15/Stocks/Data/"
        fn="raw_ts_%s_%s.xlsx" %(sdt_ds,edt_ds)
        ffn=fp+fn
        df_all.to_excel(ffn,index=False)
        print("Download data saved as file: %s" %(ffn) )
    return df_all


def stk_daily_hist_old(_sdt,_edt=None,save=True):
    if _edt==None:
        _edt=_sdt
        
    dt_tmp = str(_sdt).replace('-','')
    if not(type(int(dt_tmp))==int and len(str(dt_tmp))==8 and str(dt_tmp)[:3] == '201'):
        print("Date format not recognized. Please use yyyymmdd or yyyy-mm-dd as input.")
        return None
    dtt = datetime.datetime.strptime(str(dt_tmp),"%Y%m%d").date()
    dtf = dtt.strftime("%Y-%m-%d")
    sdtfs= dt_tmp      
    sdt = dtf    
    
    dt_tmp = str(_edt).replace('-','')
    if not(type(int(dt_tmp))==int and len(str(dt_tmp))==8 and str(dt_tmp)[:3] == '201'):
        print("Date format not recognized. Please use yyyymmdd or yyyy-mm-dd as input.")
        return None
    dtt = datetime.datetime.strptime(str(dt_tmp),"%Y%m%d").date()
    dtf = dtt.strftime("%Y-%m-%d")
    edtfs= dt_tmp      
    edt = dtf 

    df_all=pd.DataFrame()
    stk_info=ts.get_stock_basics().sort_index()  
    lp_cnt=len(stk_info)
    lp_cnt= 100
    s_time = datetime.datetime.now()
    for i in range(50,lp_cnt):
        if i%(lp_cnt//10)==0:
            print("Completed: " + str(round(i*100/lp_cnt,0))+'%')
            wm.time_estimate(s_time,i,lp_cnt)
        stk_id = stk_info.index[i]
        stk_nm=stk_info.name.iloc[i]
        try:
            ts.get_hist_data(stk_id,start=sdt,end=edt)            
        except:
            continue
        try:
            ts.get_h_data(stk_id,start=sdt,end=edt,autype='None')            
        except:
            continue
        print(str(stk_id))
        df2 = ts.get_hist_data(stk_id,start=sdt,end=edt)
        if len(df2)==0:
            continue
        df1 = ts.get_h_data(stk_id,start=sdt,end=edt,autype='None')        
        if df1 is None or df2 is None:
            continue
        df2.drop(['volume','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20'],axis=1,inplace=True)
        df=pd.merge(df2,df1[['volume','amount']],left_index=True,right_index=True)
        if list(stk_id)[0]==6:
            mkt='.SH'
        else:
            mkt='.SZ'
        df['sid']=stk_id
        df['code']=list(map(lambda x: x+'.SH' if x[0]==6 else(x+'.SZ'),df.sid))
        df['name']=stk_nm
        df.rename(columns={'turnover':'turn','p_change':'pctchange','price_change':'change'},inplace=True)
        df['preclose']=round(df.close*100/(100+df.change),2)
        df['amplitude'] = (df.high-df.low)/df.preclose
        df['highlimit']=list(map(lambda x,y: '是' if x >=round(y*1.1,2) else('否'), df.close,df.preclose))
        df['lowlimit']=list(map(lambda x,y: '是' if x <=round(y*0.9,2) else('否'), df.close,df.preclose))
        df=df.reset_index()
        df['month']=df['date'].astype(str).str[:10].str.replace('-','').astype(int)
        #df['month'].astype(int)
        df_all=df_all.append(df.drop(['date'],axis=1))
    fp = "/Users/wyatt/Documents/Docs-wMBP15/Stocks/Data/"
    fn="daily_raw_hist_" + str(sdtfs) + "_" + str(edtfs) + ".xlsx"
    ffn=fp+fn
    if save==True:
        df_all.to_excel(ffn,index=False)
        print("Download data saved as file: %s" %(ffn) )
    return df_all

def stk_info_update(show=True,save=True):
    df=ts.get_stock_basics().reset_index().sort_values('code').reset_index(drop=True)
    lp_cnt = len(df)    
    fp = "/Users/wyatt/Documents/Docs-wMBP15/Stocks/Data/"
    ffn=fp + "Stock_Info.xlsx"
    df.to_excel(ffn,index=False)
    if show==True:
        print("%s stocks info loaded." %(lp_cnt))
        if save==True:
            print("Stock Info file save as: %s" %(ffn))
    return df
        
def stk_daily_raw(_dt,save=True):
    '''
    get all stock ohlc price
    '''
    stk_info=ts.get_stock_basics()
    lp_cnt=len(stk_info)
    ffn_stk_info=pth + "Stock_Basic_Info_" + datetime.date.today().strftime("%Y%m%d") + ".xlsx"
    if not os.path.exists(ffn_stk_info):
        stk_info.sort_index().to_excel(ffn_stk_info)
    _edt = datetime.datetime.strptime(str(_dt),"%Y%m%d").date()
    _sdt = _edt - datetime.timedelta(20)
    _sdts= _sdt.strftime("%Y-%m-%d")
    _edts= _edt.strftime("%Y-%m-%d")
    tmp = ts.get_k_data("000001",start=_sdts,end=_edts,index="Y").sort_values(by='date',ascending=False)
    _edts = tmp.iat[0,0]
    _sdts = tmp.iat[4,0] 
    _sdts = _edts
    tt_all=pd.DataFrame()
    s_time = datetime.datetime.now()
    for i in range(lp_cnt):
        if i%(lp_cnt//10)==0:
            print("Completed: " + str(round(i*100/lp_cnt,0))+'%')
            wm.time_estimate(s_time,i,lp_cnt)
        stk_id = stk_info.sort_index().index[i]
        tt_tmp = ts.get_k_data(stk_id,start=_sdts,end=_edts)
        if len(tt_tmp)==0:
            continue
        tt_all=tt_all.append(tt_tmp)
    if save==True:
        tt_all.set_index('code').to_excel(pth + _edts +".xlsx")
    print("Data for: " + str(_dt) + " download completed.")
    return tt_all
  
def stk_5minutes_raw(dt):
    """ This can get the stock's specify date's detail transaction by minutes.
    This function will only return one day's tick data at a time. """
    stk_info=ts.get_stock_basics().iloc[:,0].sort_index()
    lp_cnt=len(stk_info)
    #lp_cnt=10 #limit stock # to control for testing
    dtf = datetime.datetime.strptime(str(dt),"%Y%m%d").strftime("%Y-%m-%d") 
    tt_all=pd.DataFrame()
    s_t=datetime.datetime.now()
    for i in range(lp_cnt):
        print(i,end='')
        if i%(lp_cnt//10)==0:
            print("Completed: " + str(i) + str(round(i*100/lp_cnt,0))+'%')
            wm.time_estimate(s_t,i,lp_cnt)
        stk_id = stk_info.index[i]
        try:
            tt_tmp = ts.get_tick_data(stk_id,date=dtf,src='tt')
        except:
            continue
        else:
            if tt_tmp is None:
                continue
            tt_tmp['code'] = stk_id
            tt_tmp['lowest'] = list(map(lambda x: 1 if x == min(tt_tmp['price']) else(0), tt_tmp['price']))
            tt_tmp['highest'] = list(map(lambda x: 1 if x == max(tt_tmp['price']) else(0), tt_tmp['price']))
            df=tt_tmp[['time','lowest','highest']][(tt_tmp['lowest']==1)|(tt_tmp['highest']==1)]
            #df=tt_tmp[['time','lowest','highest']].groupby('time').sum() 
            tt_all=tt_all.append(df)
    #print(tt_all)
    #tt_all['date']=dtf
    tt_all['timef']=tt_all['time'].str[:5]
    tt_all=tt_all[['timef','lowest','highest']].groupby('timef').sum()        
    tt_all.sort_index().to_excel(pth+ "minutes_" + str(dt) + ".xlsx")
    #tt_all.set_index('code')[['date','amount']].to_excel(pth+"amtt.xlsx") 
    print("Data for: " + str(dtf) + " download completed.")
    
def stk_5minutes_raw2():
    """ This can get the stock's past 14 days' 5-min tick data, if the tick interval in longer, like 15/30/60, the more day's tick data will be return.
    This function do not accetp date range. """
    stk_info=ts.get_stock_basics()
    lp_cnt=len(stk_info)
    #lp_cnt = 50
    #dtf = datetime.datetime.strptime(str(dt),"%Y%m%d").strftime("%Y-%m-%d") 
    tt_all=pd.DataFrame()
    for i in range(lp_cnt):
        if i%(lp_cnt//4)==0:
            print("Completed: " + str(round(i*100/lp_cnt,0))+'%')
        stk_id = stk_info.sort_index().index[i]
        df = ts.get_k_data(stk_id,ktype='5')
        if df is None:
            continue
        #tt_tmp['code'] = stk_id
        df=df.loc[:,['date','high','low']]
        df['dt']=df['date'].str[:10]
        df['time']=df['date'].str[11:]
        df['lowest']=list(map(lambda x,y: 1 if x==x and y == list(df[['dt','low']].groupby('dt').min().loc[x]) else 0,df['dt'],df['low']))
        df['highest']=list(map(lambda x,y: 1 if x==x and y == list(df[['dt','high']].groupby('dt').max().loc[x]) else 0,df['dt'],df['high']))
        df=df[['time','lowest','highest']][(df['lowest']==1)|(df['highest']==1)]
        tt_all=tt_all.append(df)
    #print(tt_all)
    tt_all=tt_all[['time','lowest','highest']].groupby('time').sum()        
    tt_all.sort_index().to_excel(pth+"minutes_k.xlsx")
    

def fjb(dsn,grp=False):
    """
    input df should have three columns: price, volume, amount
    """
    if sum(dsn.columns.isin(['volume','price','amount'])*1) !=3:
        print("Input data should have these three columns: price, volume, amount")
    else:
        #tmp=dsn.copy()    
        tmp=dsn[['price','volume','amount']].groupby(by='price').sum().sort_values('volume',ascending=False)    
        tmp['pct_vol']=round(tmp.volume*100/tmp.volume.sum(),1)
        tmp['pct_amt']=round(tmp.amount*100/tmp.amount.sum(),1)#.astype(int)
        tmp['pct_gap'] = tmp.pct_amt - tmp.pct_vol
        tmp['sid'] = dsn.code.iloc[0]
        tmp['name'] = dsn.name.iloc[0]
        tmp['date'] = dsn.datef.iloc[0]
        #tmp['pct_amt_cum']=round(tmp.pct_amt.cumsum(),0).astype(int)
        if grp:
            tmp_grp=tmp.copy().reset_index().sort_values('price',ascending=False).reset_index(drop=True)
            tmp_grp['grp']=tmp_grp.index*10//len(tmp_grp)
            tmp_grp1=tmp_grp[['grp','price']].groupby('grp').mean()
            tmp_grp2=tmp_grp[['grp','volume','amount','pct_vol','pct_amt','pct_gap']].groupby('grp').sum()
            tmp_g = pd.concat([tmp_grp1,tmp_grp2],axis=1)
            tmp_g['price']=round(tmp_g.price,2)
            tmp_g.rename(columns={'price':'price_avg'},inplace=True)
            tmp_g.drop(['volume','amount','pct_gap','pct_amt'],axis=1,inplace=True)
            return tmp_g
        else:
            return tmp.drop(['volume','amount','pct_gap','pct_amt'],axis=1)

    
def price_line(dsn,x='timef',y='close',show=True):
    #from matplotlib.ticker import MultipleLocator, FormatStrFormatter    
    
    tmp = dsn.reset_index()
    tmp.sort_values(x)
    fig,ax = plt.subplots(figsize=(15,6))  
    
    x_ticks = [t for t in range(len(tmp)) if t%10==0]
    x_labels = [tmp[x].iloc[t] for t in range(len(tmp)) if t%10==0]
    #ax.xaxis.set_major_locator(MultipleLocator(50)) 
    #ax.xaxis.set_ticklabels(x_labels)
    sid = tmp['code'][0]
    sid_nm = tmp['name'][0]
    dtf=tmp['datef'][0]
    ttxt = 'Minute Price Line for %s(%s) on %s' %(sid,sid_nm,dtf)
    plt.title(ttxt,fontsize=30,fontproperties=chnchr())
    plt.xticks(x_ticks, x_labels, rotation=0) 
    plt.plot(range(len(tmp)),tmp[y]) 
    #plt.grid(axis='x')  
    if show==True:
        plt.show()  
    return fig


def price_bar(dsn,x='price',y='pct_vol',save=False,show=True):
    tmp = dsn.reset_index()
    xx = list(tmp[str(x)])# Make an array of x values
    yy = list(tmp[str(y)])# Make an array of y values for each x value
    x_label=xx.copy()
    fig=plt.figure(figsize=(10,min(len(xx),6)))
    #fig=plt.figure()
    plt.barh(xx,yy,height=0.01,label='price pressure')
    # numpy.random.uniform(low=0.0, high=1.0, size=None), normal
    #uniform均匀分布的随机数，normal是正态分布的随机数，0.5-1均匀分布的数，一共有n个
    #Y1 = np.random.uniform(0.5,1.0,n)
    #Y2 = np.random.uniform(0.5,1.0,n)
    #plt.bar(X,Y1,width = 0.35,facecolor = 'lightskyblue',edgecolor = 'white')
    #width:柱的宽度
    #plt.barh(xx,yy,height=0.01,label='price pressure',facecolor = 'lightskyblue',edgecolor = 'white')
    #plt.barh(xx,yy,height=0.01,label='price pressure',fc = 'lightskyblue',ec = 'white')
    #plt.bar(X+0.35,Y2,width = 0.35,facecolor = 'yellowgreen',edgecolor = 'white')
    #水平柱状图plt.barh，属性中宽度width变成了高度height
    #打两组数据时用+
    #facecolor柱状图里填充的颜色
    #edgecolor是边框的颜色
    #想把一组数据打到下边，在数据前使用负号
    #plt.bar(X, -Y2, width=width, facecolor='#ff9999', edgecolor='white')
    #给图加text
    '''
    for x,y in zip(X,Y1):
        plt.text(x+0.3, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

        for x,y in zip(X,Y2):
            plt.text(x+0.6, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
            plt.ylim(0,+1.25)
    '''
    plt.grid(axis='x')
    plt.axis('auto')

    #plt.legend('best')
    plt.xlabel('pct_vol')
    plt.ylabel('price')
    title = "%s(%s)分价表_%s" %(tmp.sid[0],tmp.name[0],tmp.date[0])
    plt.title(title,fontproperties=chnchr())
    if save==True:
        fig.savefig('/Users/wyatt/Documents/Docs-wMBP15/Stocks/single/'+title+'.png')
    if show==True:
        plt.show()


def stk_txn_raw(sid,dt=datetime.date.today().strftime("%Y%m%d")):
    """ 
    sid: will return this stock's minutes detail transactions on dt
    dt: the single day to return the minutes detail transactions, default the most recent trade date
    """
    if ts.get_stock_basics().index.isin([str(sid)]).max() != True:
        print("Stock ID not recognized, please check.")
    else:
        dt_tmp = str(dt).replace('-','')
        if type(int(dt_tmp))==int and len(str(dt_tmp))==8 and str(dt_tmp)[:3] == '201':
            dtt = datetime.datetime.strptime(str(dt_tmp),"%Y%m%d").date()
            dtf = dtt.strftime("%Y-%m-%d")
            dtfs= dt_tmp
            sid_nm = ts.get_stock_basics().loc[sid,'name']
            df=pd.DataFrame()
            df = ts.get_tick_data(sid,date=dtf,src='tt')
            df['lowest'] = list(map(lambda x: 1 if x == min(df['price']) else(0), df['price']))
            df['highest'] = list(map(lambda x: 1 if x == max(df['price']) else(0), df['price']))
            df['date']= dtt
            df['datef']=dtf
            df['datefs']=dt_tmp
            df['timef']=df['time'].str[:5]
            df['chg_pct']=df.change/df.price
            df['vol_pct']=df.volume/df.volume.sum()
            df['amt_pct']=df.amount/df.amount.sum()
            df['code']=sid
            df['name']=sid_nm
            df['dt']=df.datef + ' ' +df.time
            df.dt=pd.to_datetime(df.dt)
            df.set_index('dt',inplace=True)
            df.sort_index(inplace=True)
            #df.sort_index().to_excel(pth+ "minutes_" + str(dt) + ".xlsx")
            #tt_all.set_index('code')[['date','amount']].to_excel(pth+"amtt.xlsx") 
            print("Data for: %s(%s) on date %s download completed." %(sid,sid_nm,dtf))
            return df
        else:
            print("Date format not recognized. Please use yyyymmdd or yyyy-mm-dd as input.")
    
def stk_txn_grp(dsn,grp='T'):
    '''
    grp: 
        'T' to grp in minutes
        '5T' to grp in 5 minutes
        '2H' to grp in 2 hours
    '''
    tt=dsn.copy()
    #grp=grp.str.replace('
    tt1 = tt.price.resample(grp).ohlc()
    tt2=tt[['change','volume','amount']].resample(grp).sum()
    tt3=tt[['highest','lowest','date','datef','datefs','timef','code','name']].resample(grp).max()
    df = pd.concat([tt1,tt2,tt3],axis=1)
    df['change'] = round(df['change'],2)
    df['chg_pct']=df.change/(df.close+df.change)
    df['vol_pct']=df.volume/df.volume.sum()
    df['amt_pct']=df.amount/df.amount.sum()
    df=df[df.close.notnull()]
    return df

def stk_minutes_ndays(sid,dt=datetime.date.today().strftime("%Y%m%d"),dcnt=3):
    '''
    sid: stock id
    dt: yyyymmdd or yyyy-mm-dd, from this date, past dcnt days' minutes detail data will be return
    dcnt: identify how many days' minutes detail data to return, default past 3 days
    '''
    if ts.get_stock_basics().index.isin([str(sid)]).max() != True:
        print("Stock ID not recognized, please check.")
    else:
        dt_tmp = str(dt).replace('-','')
        if not (type(int(dt_tmp))==int and len(str(dt_tmp))==8 and str(dt_tmp)[:3] == '201'):
            print("Data format not recognized. Please use yyyymmdd or yyyy-mm-dd as input.")
        else:
            d_cnt=0
            df_all=pd.DataFrame()
            for d in range(0,dcnt*3):
                dtt=datetime.datetime.strptime(str(dt_tmp),"%Y%m%d")+datetime.timedelta(-d)
                dtf=dtt.strftime("%Y-%m-%d")
                dtfs=dtt.strftime("%Y%m%d")
                if d_cnt < dcnt and dtt.weekday()+1 <=5:
                    d_cnt+=1
                    df = stk_txn_raw(sid,dtf)
                    df['day_cnt']=d_cnt
                    df_all=df_all.append(df)
            print("All data download completed.")
            return df_all

def fjb_ndays(dsn,dcnt=3):
    '''
    dsn: input data should have detail minute transactions and day_cnt
    dcnt: to return how many days' price bar chart in a time, default/max past 3 days
    '''
    if dcnt>3 or dcnt<1:
        print("Max dcnt is 3, please reset or quit.")
        return None
    if sum(dsn.columns.isin(['day_cnt'])*1) !=1:
        print("Input data should have these column(s): day_cnt")
        return None
    else:   
        df_all = dsn.copy()
        fjb1=fjb(df_all[df_all.day_cnt==1])#.add_suffix('_d1')
        #fjb2=fjb(df_all[df_all.day_cnt==2])#.add_suffix('_d2')
        #fjb3=fjb(df_all[df_all.day_cnt==3])#.add_suffix('_d3')
        fjb12=fjb(df_all[df_all.day_cnt<=2])#.add_suffix('_d12')
        fjb13=fjb(df_all[df_all.day_cnt<=3])#.add_suffix('_d13')
        #fjb1_grp=fjb(df_all[df_all.day_cnt==1],True)#.add_suffix('_d1')
        #fjb12_grp=fjb(df_all[df_all.day_cnt<=2],True)#.add_suffix('_d12')
        #fjb13_grp=fjb(df_all[df_all.day_cnt<=3],True)#.add_suffix('_d13')
        ff=pd.concat([fjb1,fjb12.add_suffix('_d12'),fjb13.add_suffix('_d13')],axis=1).sort_values('pct_vol',ascending=False)
        #ff1=pd.concat([fjb1,fjb12,fjb13])
        #ff1.sort_values('pct_vol',ascending=False)
        #ff2=ff1.groupby('price').mean()
        #price_bar(fjb1,'price','pct_vol',str(sid) + '_'+str(dt) +'_day1')
        #price_bar(fjb12,'price','pct_vol',str(sid) + '_'+str(dt) +'_day12')
        #price_bar(fjb13,'price','pct_vol',str(sid) + '_'+str(dt) +'_day13')
        #price_bar(ff2,title=str(sid) + '_'+str(dt) +'_day13combine')
        
        sid = df_all.code.iloc[0]
        sid_nm = df_all.name.iloc[0]
        dt = df_all.date.iloc[0]
        #plt.figure(1)#创建图表1  
        fig=plt.figure(figsize=(10,dcnt*7))
        plt.figure(1)#创建图表2  
        
        #names=()
        #for lp in range(1,dcnt+1):
        #    names['ax%s' %lp] = plt.subplot(dcnt,1,lp)
                
        ax1=plt.subplot(dcnt,1,1)#在图表2中创建子图1  
        plt.sca(ax1) 
        gap1 = fjb1.index.max()-fjb1.index.min()
        gap2=max(0.01,round((gap1)/20,2))
        ax1.yaxis.set_major_locator(MultipleLocator(gap2))  #将x主刻度标签设置为gap2的倍数
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%5.2f')) #设置x轴标签文本的格式
        plt.barh(list(fjb1.index),list(fjb1['pct_vol']),height=gap2) 
        plt.grid(axis='x')
        title = str("%s(%s)_%s_" %(sid,sid_nm,str(dt)))
        plt.title(title+'d1',fontproperties=chnchr())
        
        if dcnt>=2:
            ax2=plt.subplot(dcnt,1,2)#在图表2中创建子图2             
            plt.sca(ax2)  
            ax2.yaxis.set_major_locator(MultipleLocator(gap2))  #将x主刻度标签设置为gap2的倍数
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%5.2f')) #设置x轴标签文本的格式
            plt.barh(list(fjb12.index),list(fjb12['pct_vol']),height=gap2)
            plt.grid(axis='x')
            plt.title(title+'d12',fontproperties=chnchr())
        
        if dcnt == 3:
            ax3=plt.subplot(dcnt,1,3)#在图表2中创建子图2  
            plt.sca(ax3)  
            ax3.yaxis.set_major_locator(MultipleLocator(gap2))  #将x主刻度标签设置为gap2的倍数
            ax3.yaxis.set_major_formatter(FormatStrFormatter('%5.2f')) #设置x轴标签文本的格式
            plt.barh(list(fjb13.index),list(fjb13['pct_vol']),height=gap2)
            plt.grid(axis='x')
            plt.title(title+'d13',fontproperties=chnchr())
        
        ffn = '/Users/wyatt/Documents/Docs-wMBP15/Stocks/single/%s(%s)_%s.png' %(str(sid),sid_nm,str(dt))
        print("Graphs will be saved as: \n" + ffn)
        fig.savefig(ffn)
        plt.show()  

        return ff