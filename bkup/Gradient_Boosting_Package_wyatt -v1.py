"""Gradient Boosting Learner in AMH

    This model inherits sklearn.ensemble.GradientBoostingClassifier by 
    incoporating C-stat, Maximum KS & decile lift which are most impor-
    tant statistics to validate model prediction for campaign targeting
    (order ranking).

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.artist
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.ticker import Formatter, FixedLocator

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier

from IPython.display import clear_output
from datetime import datetime
import pytz

class GradientBoostingLearner_Shawn(GradientBoostingClassifier):
    def __init__(self,loss='deviance',learning_rate=0.1,n_estimators=100,subsample=1.0,criterion='friedman_mse',min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.,max_depth=3,min_impurity_split=1e-7,init=None,random_state=None,max_features=None,verbose=0,max_leaf_nodes=None,warm_start=False,presort='auto',df_in=None,str_group=None,str_score=None,str_resp=None,fmt_count=None,fmt_group=None,fmt_resp=None,fmt_resp_rate=None,fmt_rand_rate=None):
        super(GradientBoostingClassifier, self).__init__(loss=loss,
                                                         learning_rate=learning_rate, 
                                                         n_estimators=n_estimators,
                                                         criterion=criterion, 
                                                         min_samples_split=min_samples_split,
                                                         min_samples_leaf=min_samples_leaf,
                                                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                         max_depth=max_depth, 
                                                         init=init, 
                                                         subsample=subsample,
                                                         max_features=max_features,
                                                         random_state=random_state, 
                                                         verbose=verbose,
                                                         max_leaf_nodes=max_leaf_nodes,
                                                         min_impurity_split=min_impurity_split,
                                                         warm_start=warm_start,
                                                         presort=presort)
        '''self.df_in=df_in'''
        self.str_score = 'score'
        self.str_resp=str_resp
        self.str_group = 'group'
        self.fmt_resp = 'response'
        self.fmt_group='group'
        self.fmt_resp_rate='response_rate'
        self.fmt_rand_rate='random_select'
        self.fmt_count='cust_cnt'
        
    def decile_lift(self, df_scored):
        '''
        df_in: a dataframe that contains both group & response columns
        str_group: a string that specifies group name
        str_resp: a string that specifies response name
        '''
        if df_scored is None:
            print("Error: no scored file for decile_lift() !!!!")   

        else:
            # group by decile
            deciles = df_scored.groupby([self.str_group]).agg({self.str_resp: [np.size, np.mean]})
            deciles.columns=deciles.columns.droplevel(level=0)

            deciles = deciles.reset_index(drop=False)
            deciles['random_select'] = df_scored[self.str_resp].mean()

            deciles.rename(columns = {'size':'cust_cnt','mean':'response_rate'}, inplace=True)
            deciles['lift'] = deciles['response_rate'] / deciles['random_select']


            # deciles['rand']=df_scored[self.str_resp].mean()
            # deciles['lift']=deciles['mean']/df_scored[self.str_resp].mean()
            # deciles['group_num']=np.arange(1,11,1)

            deciles_record = deciles.reset_index(drop=True)

            deciles_record['cnt_name'] =str('decile_cnt_')+ deciles_record.group.astype(str)
            deciles_record['rr_name'] =str('decile_rr_')+ deciles_record.group.astype(str)
            deciles_record['lift_name']=str('decile_lift_')+ deciles_record.group.astype(str)

            deciles_cnt= pd.DataFrame(deciles_record['cust_cnt'].values,index=deciles_record.cnt_name).T
            deciles_cnt_new = deciles_cnt.reset_index(drop=True)

            deciles_rr= pd.DataFrame(deciles_record['response_rate'].values,index=deciles_record.rr_name).T
            deciles_rr_new = deciles_rr.reset_index(drop=True)

            deciles_lift= pd.DataFrame(deciles_record['lift'].values,index=deciles_record.lift_name).T
            deciles_lift_new = deciles_lift.reset_index(drop=True)
            deciles_score = pd.concat([deciles_cnt_new,deciles_rr_new,deciles_lift_new], axis=1)

        return deciles_score, deciles
    
    
    def maximum_ks(self, df_scored):
        # Revised by Shawn on 20 Jun 2017
        """
            df_in: a dataframe that contains both group & response columns
            str_score: a string that specifies score name
            str_resp: a string that specifies response name
        """
        if df_scored is None:
            print("Error: no scored file for for maximum_ks() !!!!")
        else:
            # Max KS
            max_ks_sort=df_scored.sort_values([self.str_score],ascending=False)
            max_ks_sort['response']=df_scored[self.str_resp]
            max_ks_sort.index=range(1,len(max_ks_sort)+1)
            max_ks_sort['cum_good']=max_ks_sort.response.cumsum()
            max_ks_sort['cum_bad']=max_ks_sort.index - max_ks_sort.cum_good
            max_ks_sort['cum_good_rate'] = max_ks_sort.cum_good / max_ks_sort.response.sum()
            max_ks_sort['cum_bad_rate']= max_ks_sort.cum_bad / (max_ks_sort.response.size-max_ks_sort.response.sum())
            max_ks_sort['cum_rand_rate']= max_ks_sort.index / max_ks_sort.response.size
            max_ks_sort['ks'] = max_ks_sort.cum_good_rate - max_ks_sort.cum_bad_rate

            max_ks_score = max_ks_sort[(max_ks_sort.ks==max_ks_sort.ks.max())]
            max_ks_score = max_ks_score[['cum_good_rate','cum_rand_rate','ks']]
            max_ks_score.rename(columns={'cum_rand_rate':'max_ks_pop','ks':'max_ks'}, inplace=True)
            max_ks_score = max_ks_score.reset_index(drop=True)

        return max_ks_score, max_ks_sort
    
    def c_stat(self, df_scored):
        # Revised by Shawn on 20 Jun 2017
        """
            df_in: a dataframe that contains both group & response columns
            str_score: a string that specifies score name
            str_resp: a string that specifies response name
        """
        if df_scored is None:
            print("Error: no scored file for c_stat() !!!!")
        else:
            # C-stat / concordant %
            c_stat_sort=df_scored.sort_values([self.str_score],ascending=True)
            c_stat_sort['response']=df_scored[self.str_resp]
            c_stat_sort=c_stat_sort.reset_index(drop=True)
            c_stat_sort['rp']=c_stat_sort.index
            num_resp=c_stat_sort.response.sum()
            rp_resp_sum=sum(c_stat_sort.response*c_stat_sort.rp)
            row_count=c_stat_sort.response.count()
            c_stat=(rp_resp_sum-0.5*num_resp*(num_resp-1))/(num_resp*(row_count-num_resp))
            c_stat_score=pd.DataFrame([c_stat],columns=['c_stat'])

        return c_stat_score
    
    def rank_decile(self, df_scored):
        # to evenly rank scored base into decile
        if df_scored is None:
            print("Error: no scored file for c_stat() !!!!")
        else:
            df_ranked = pd.DataFrame(pd.qcut(df_scored[self.str_score], 10, labels=False))
            df_ranked.rename(columns = {self.str_score : self.str_group}, inplace=True)
            df_ranked[self.str_group] = 10 - df_ranked[self.str_group]
            df_ranked_scored = pd.concat([df_scored, df_ranked], axis=1)

        return df_ranked_scored
    
    def get_result(self, df_scored, int_iter_cnt):
        lifts, lift_chart = self.decile_lift(df_scored)
        maxks, lorz_curve = self.maximum_ks(df_scored)
        cstat= self.c_stat(df_scored)

        param=pd.concat([pd.DataFrame([int_iter_cnt],columns=['model_cnt']),
                         pd.DataFrame([self.n_estimators],columns=['n_estimators']),
                         pd.DataFrame([self.learning_rate],columns=['learning_rate']),
                         pd.DataFrame([self.min_samples_split],columns=['min_samples_split']),
                         pd.DataFrame([self.min_samples_leaf],columns=['min_samples_leaf']),
                         pd.DataFrame([self.max_depth],columns=['max_depth']),
                         pd.DataFrame([self.max_features],columns=['max_features']),
                         pd.DataFrame([self.subsample],columns=['subsample']),
                         pd.DataFrame([self.random_state],columns=['random_state']),
                         pd.DataFrame([self.criterion],columns=['criterion'])
                        ], axis=1)

        model_kpi=pd.concat([pd.DataFrame([int_iter_cnt],columns=['model_cnt']), cstat, maxks, lifts], axis=1)

        return param, model_kpi
    
    # Define plot
    def lift_chart(self, df_deciles):
        grp = df_deciles[self.fmt_group]
        drr = df_deciles[self.fmt_resp_rate]
        rrr = df_deciles[self.fmt_rand_rate]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1,=ax.plot(grp,drr,marker='o',color='blue', lw=1.5)
        line2,=ax.plot(grp,rrr,ls='dashed',color='gray',lw=1.5)

        xtext = ax.set_xlabel('Deciles')
        ytext = ax.set_ylabel('% Resp')

        ax.set_xlim(0.5,10.5)
        ax.set_xticks(range(1,11,1))

        ax.set_ylim(0, max(drr)*(1.1))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda t1, _: '{:.0%}'.format(t1)))

        txt1=str("Decile 1 R.R.: " + '{:.1%}'.format(drr.loc[0]) + "; Lift: " + '{:3.2}'.format(drr.loc[0]/rrr.loc[0]))
        txt2=str("Decile 2 R.R.: " + '{:.1%}'.format(drr.loc[1]) + "; Lift: " + '{:3.2}'.format(drr.loc[1]/rrr.loc[1]))
        txt3=str("Decile 3 R.R.: " + '{:.1%}'.format(drr.loc[2]) + "; Lift: " + '{:3.2}'.format(drr.loc[2]/rrr.loc[2]))

        plt.text(grp[1]-0.7, drr.loc[0]*0.99, txt1, ha='left', rotation=0, wrap=True)
        plt.text(grp[2]-0.7, drr.loc[1]*0.99, txt2, ha='left', rotation=0, wrap=True)
        plt.text(grp[3]-0.7, drr.loc[2]*0.99, txt3, ha='left', rotation=0, wrap=True)

        plt.legend((line1, line2), ('Model', 'Random'), loc='upper right', bbox_to_anchor=[0.95,0.95], shadow=True)

        plt.suptitle('Lift Chart', fontsize=14)
        plt.show()

        return
    
    def lorenz_curve(self, df_kstable, df_maxks):
        lorenz=pd.DataFrame({'cum_good':df_kstable.cum_good_rate.values[list(range(1,len(df_kstable)+1,100))],
                             'cum_rand':df_kstable.cum_rand_rate.values[list(range(1,len(df_kstable)+1,100))],
                             'cum_bad':df_kstable.cum_bad_rate.values[list(range(1,len(df_kstable)+1,100))]})
        t0 = lorenz.cum_rand.values
        t1 = lorenz.cum_good.values
        t2 = lorenz.cum_rand.values
        t3 = lorenz.cum_bad.values

        fig = plt.figure()
        ax = fig.add_subplot(111)

        max_ks_val = df_maxks['max_ks'].values[0]
        max_ks_pop = df_maxks['max_ks_pop'].values[0]
        max_ks_cgr = df_maxks['cum_good_rate'].values[0]

        line1,=ax.plot(t0,t1,ls='solid',color='blue', lw=1.5)
        line2,=ax.plot(t0,t2,ls='dashed',color='green', lw=1.5)
        line3,=ax.plot(np.array([max_ks_pop,max_ks_pop]),np.array([0,max_ks_cgr]),ls='dashdot',color='grey',lw=1.5)
        line4,=ax.plot(np.array([0,max_ks_pop]),np.array([max_ks_cgr,max_ks_cgr]),ls='dashdot',color='grey',lw=1.5)
        mark1,=ax.plot(max_ks_pop,max_ks_cgr,marker='>',markersize=10,color='blue')

        txt=str("Max KS: " + '{:.0%}'.format(max_ks_val) + " at " + '{:.0%}'.format(max_ks_pop) + " of Population")
        plt.text(max_ks_pop + 0.03, max_ks_cgr - 0.01, txt, ha='left', rotation=0, wrap=True)

        xtext = ax.set_xlabel('% Cumulative Population')
        ytext = ax.set_ylabel('% Cumulative Good')

        ax.set_xlim(0.,1.)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda t0, _: '{:.0%}'.format(t0)))

        ax.set_ylim(0.,1.)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda t1, _: '{:.0%}'.format(t1)))

        plt.legend((line1, line2), ('Model', 'Random'), loc='lower right', bbox_to_anchor=[0.95,0.1], shadow=True)

        plt.suptitle('Lorenz Curve', fontsize=14)
        plt.show()

        return
        

class GBM_Shawn():
    def __init__(self,mode=None,df_train=None,df_valid=None,str_resp=None,str_id=None):
        self.mode=mode
        self.df_train=df_train
        self.df_valid=df_valid
        self.str_resp=str_resp ## response variable
        self.str_id=str_id

        self.train_base=self.df_train ## save the original training dataframe
        self.param=pd.DataFrame() ## full set of parameters
        self.model_kpi=pd.DataFrame()
        self.importance=pd.DataFrame() ## full set of parameters
        self.best_param=pd.DataFrame() ## best parameter
        self.best_model_kpi=pd.DataFrame()
        self.best_driver=pd.DataFrame()
        self.best_model=None ## trained model with the best parameter

        self.iter_cnt=1
        
        #to grid search all parameters and return the full set of parameters and KPIs
        if self.mode is None or self.mode == 'superfast':
            self.n_estimators= [70]
            self.learning_rate= [0.1]
            self.min_samples_split= [50,100]
            self.min_samples_leaf= [25,50]
            self.max_depth= [3,4,5,6]
            self.max_features= ['sqrt']
            self.subsample=[0.9]
            self.random_state=[10]
            self.criterion= ['friedman_mse'] # mae, mse
        elif self.mode=='fast':
            self.n_estimators= [70,80,90]
            self.learning_rate= [0.1,0.2]
            self.min_samples_split= [50,100,200]
            self.min_samples_leaf= [25,50,100]
            self.max_depth= [3,4,5,6]
            self.max_features= ['sqrt']
            self.subsample= [0.9]
            self.random_state=[10]
            self.criterion= ['friedman_mse'] # mae, mse
        elif self.mode=='medium':
            self.n_estimators= [80,90,100,120]
            self.learning_rate= [0.1,0.2,0.3]
            self.min_samples_split = [0.001,0.003,0.005,0.01]
            self.min_samples_leaf= [0.0005,0.0015,0.0025,0.005]
            self.max_depth= [4,5,6,7,8]
            self.max_features= ['sqrt']
            self.subsample= [0.9]
            self.random_state= [8051]
            self.criterion= ['friedman_mse'] # mae, mse
        elif self.mode=='slow':
            self.n_estimators= [50,60,70,80,90,100]
            self.learning_rate= [0.1,0.2,0.3,0.4,0.5]
            self.min_samples_split = [50,100,200,500,1000]
            self.min_samples_leaf= [25,50,100,250,500]
            self.max_depth= [3,4,5,6,7,8,9,10]
            self.max_features= ['sqrt']
            self.subsample= [0.8,0.9]
            self.random_state= [10]
            self.criterion= ['friedman_mse'] # mae, mse
        elif self.mode=='superslow':
            self.n_estimators= [50,60,70,80,90,100]
            self.learning_rate= [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5]
            self.min_samples_split = [50,100,200,500,1000]
            self.min_samples_leaf= [25,50,100,250,500]
            self.max_depth= [3,4,5,6,7,8,9,10]
            self.max_features= ['sqrt']
            self.subsample= [0.7,0.8,0.9]
            self.random_state= [10]
            self.criterion= ['friedman_mse'] # mae, mse
        elif self.mode=='customized':
            self.n_estimators= [100]
            self.learning_rate= [0.1]
            self.min_samples_split = [50]
            self.min_samples_leaf= [100]
            self.max_depth= [8]
            self.max_features= ['sqrt']
            self.subsample= [0.9]
            self.random_state= [10]
            self.criterion= ['friedman_mse'] # mae, mse
            
        self.tot_iter = len(self.n_estimators)*len(self.learning_rate)*len(self.min_samples_split)*len(self.min_samples_leaf)*len(self.max_depth)*len(self.subsample)
        
        if self.df_train is None:
            print("Training sample is missing !")

        if self.df_valid is None:
            print("Validation sample is missing !")

        if self.str_resp is None:
            print("Response variable is not specified !")
            
    def _training(self):
        ## data preparation
        # training sample

        y_train=self.df_train[self.str_resp].values
        x_train=self.df_train.drop([self.str_resp,self.str_id], axis=1).values

        # validation sample
        y_valid=self.df_valid[self.str_resp].values
        x_valid=self.df_valid.drop([self.str_resp,self.str_id], axis=1).values


        # parameter tunning in Gradient Boosting
        for i in self.n_estimators:
            for j in self.learning_rate:
                for k in self.min_samples_split:
                    for l in self.min_samples_leaf:
                        for m in self.max_depth:
                            for n in self.max_features:
                                for o in self.subsample:
                                    for p in self.random_state:
                                        for r in self.criterion:
                                            # model training
                                            clf = GradientBoostingLearner_Shawn(n_estimators= i,
                                                                                learning_rate= j,
                                                                                min_samples_split = k,
                                                                                min_samples_leaf= l,
                                                                                max_depth= m,
                                                                                max_features= n,
                                                                                subsample= o,
                                                                                random_state= p,
                                                                                criterion= r,
                                                                                str_resp= self.str_resp
                                                                                ).fit(x_train, y_train)
                                            # Display progress
                                            if divmod(self.iter_cnt,10)[1]==0:
                                                print('Iteration: ' + str(self.iter_cnt) + ' out of ' + str(self.tot_iter))
                                            # Validation sample
                                            # to enahnce - 
                                                # 1) standardize ranking; 
                                                # 2) merge by cust_id (needed?)
                                                # 3) to save train KPIs
                                            p_valid_scored = pd.DataFrame(clf.predict_proba(x_valid))
                                            p_valid_scored['response'] = self.df_valid[self.str_resp]
                                            p_valid_scored = p_valid_scored.drop([0], axis=1)
                                            p_valid_scored.rename(columns = {1:'score'}, inplace=True)
                                            p_valid_ranked = clf.rank_decile(p_valid_scored)

                                            # get parameters and KPIs
                                            df_param, df_model_kpi = clf.get_result(p_valid_ranked, self.iter_cnt)

                                            self.param=self.param.append(df_param)
                                            self.model_kpi=self.model_kpi.append(df_model_kpi)

                                            # get driver importance
                                            df_imptc=pd.DataFrame(clf.feature_importances_,
                                                                  index=self.df_train.drop([self.str_resp,self.str_id],axis=1).var().index)
                                            df_imptc=df_imptc.T
                                            df_imptc['model_cnt']=self.iter_cnt
                                            self.importance=self.importance.append(df_imptc)

                                            # ++self.iter_cnt
                                            self.iter_cnt=self.iter_cnt+1


        best_ref = self.model_kpi[(self.model_kpi['c_stat']==self.model_kpi['c_stat'].max())]
        best_num = best_ref['model_cnt'].values[0]

        print("Best Model Candidate : Model_cnt = " + str(best_num))

        self.best_param_t=pd.DataFrame(self.param[(self.param.model_cnt==best_num)])
        self.best_param_t=self.best_param_t.reset_index(drop=True)

        self.best_param=self.best_param_t.T
        self.best_param.rename(columns={0:'Parameter Settings'}, inplace=True)
        self.best_param=self.best_param.reset_index(drop=False)

        self.best_model_kpi=pd.DataFrame(self.model_kpi[(self.model_kpi.model_cnt==best_num)]).T
        self.best_model_kpi.rename(columns={0:'Model KPI'}, inplace=True)
        self.best_model_kpi=self.best_model_kpi.reset_index(drop=False)

        self.best_driver=pd.DataFrame(self.importance[(self.importance.model_cnt==best_num)]).T
        self.best_driver.rename(columns={0:'Importance'}, inplace=True)
        self.best_driver=self.best_driver.sort_values(['Importance'], ascending=False)
        self.best_driver=self.best_driver.reset_index(drop=False)

        # save the best model
        self.best_model = GradientBoostingLearner_Shawn(n_estimators= self.best_param_t['n_estimators'].values[0],
                                                        learning_rate= self.best_param_t['learning_rate'].values[0],
                                                        min_samples_split = self.best_param_t['min_samples_split'].values[0],
                                                        min_samples_leaf= self.best_param_t['min_samples_leaf'].values[0],
                                                        max_depth= self.best_param_t['max_depth'].values[0],
                                                        max_features= self.best_param_t['max_features'].values[0],
                                                        subsample= self.best_param_t['subsample'].values[0],
                                                        random_state= self.best_param_t['random_state'].values[0],
                                                        criterion= self.best_param_t['criterion'].values[0],
                                                        str_resp= self.str_resp).fit(x_train, y_train)
        return
    
    def _validating(self, df_to_valid):
        # validation sample
        #y_valid=df_to_valid[self.str_resp].values
        x_valid=df_to_valid.drop(self.str_resp, axis=1).values

        p_predict=pd.DataFrame(self.best_model.predict_proba(x_valid))
        p_actual=pd.DataFrame(df_to_valid[self.str_resp])
        p_valid=pd.concat([p_predict, p_actual], axis=1)
        p_valid.rename(columns = {0:'score_0',1:'score_1'}, inplace=True)
        p_rank=pd.DataFrame(pd.qcut(p_valid['score_1'], 10, labels=False))
        p_rank.rename(columns = {'score_1':'group'}, inplace=True)
        p_valid2=pd.concat([p_valid, p_rank], axis=1)

        # get parameters and KPIs
        df_param=self.best_model.get_result(p_valid2)

        return df_param

    def _modeltest(self, df_to_test):
        if df_to_test is None:
            print("No scoring file is specified !!!")
            return
        elif len(df_to_test.index)==0:
            print("No records found from scoring file !!!")
            return
        else:
            df_id= pd.DataFrame(df_to_test[self.str_id])
            df_resp = pd.DataFrame(df_to_test[self.str_resp])
            # to_score = df_to_score.drop([self.str_id], axis=1)
            scored_test = pd.DataFrame(self.best_model.predict_proba(df_to_test.drop([self.str_resp,self.str_id], axis=1).values))
            scored_test['response'] = df_to_test[self.str_resp]
            scored_test = scored_test.drop([0], axis=1)
            scored_test.rename(columns = {1:'score'}, inplace=True)

            ranked_test = self.best_model.rank_decile(scored_test)

            # print(ranked_test)

            test_lifts, test_lift_chart = self.best_model.decile_lift(ranked_test)
            test_maxks, test_lorz_curve = self.best_model.maximum_ks(ranked_test)
            test_cstat= self.best_model.c_stat(ranked_test)

            print(test_lift_chart)

            self.best_model.lift_chart(test_lift_chart)
            self.best_model.lorenz_curve(test_lorz_curve, test_maxks)

        return