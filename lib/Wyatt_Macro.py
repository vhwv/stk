import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import FontManager, FontProperties 

import datetime,os,sys
sys.path.append("sers/wyatt/Documents/Docs-wMBP15/SAS/stk_model/python/lib")


def time_estimate(s_time,lp_now,lp_ttl):
    if lp_now>=2:
        now_time = datetime.datetime.now()
        ttl_time = (now_time-s_time).seconds
        now_lp=lp_now
        ttl_lp=lp_ttl
        left_lp=ttl_lp-now_lp+1
        avg_time = ttl_time/(now_lp-1)
        left_time=avg_time*left_lp
        left_min,left_sec=divmod(left_time,60)
        left_hour,left_min=divmod(left_min,60)
        left_day,left_hour=divmod(left_hour,24)
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=left_time)
        #print("Estimated Time Left: %d day, %d hours, %02d minutes, %02d seconds" %(left_day,left_hour,left_min,left_sec))
        #print("Estimated Finish Time: %s" %finish_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Estimated Time Left: %02dh-%02dm-%02ds" %(left_hour,left_min,left_sec) + ", Estimated Time Finished: %s" %finish_time.strftime("%Y-%m-%d %H:%M:%S"))
        
def time_estimate2(lp_now,lp_ttl):
    global s_time,lp_first
    try:
        s_time
    except:
        #global s_time
        s_time = datetime.datetime.now()
        lp_first = lp_now
    if lp_now>lp_first:
        now_time = datetime.datetime.now()
        ttl_time = (now_time-s_time).seconds
        now_lp=lp_now
        ttl_lp=lp_ttl
        left_lp=ttl_lp-now_lp+1
        avg_time = ttl_time/(now_lp-lp_first)
        left_time=avg_time*left_lp
        left_min,left_sec=divmod(left_time,60)
        left_hour,left_min=divmod(left_min,60)
        left_day,left_hour=divmod(left_hour,24)
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=left_time)
        #print("Estimated Time Left: %d day, %d hours, %02d minutes, %02d seconds" %(left_day,left_hour,left_min,left_sec))
        #print("Estimated Finish Time: %s" %finish_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Estimated Time Left: %02dh-%02dm-%02ds" %(left_hour,left_min,left_sec) + ", Estimated Time Finished: %s" %finish_time.strftime("%Y-%m-%d %H:%M:%S"))
    if lp_now==lp_ttl:
        del s_time,lp_first