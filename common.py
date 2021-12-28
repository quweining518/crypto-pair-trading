# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:27:04 2021

@author: Admin
"""

import pandas as pd
import numpy as np

data = pd.read_excel('成立以来模拟组合.xlsx', sheet_name='成立以来净值', index_col=0).iloc[:205,:]
data_return = data.apply(lambda x: x.pct_change())



def Annualized_Returns(arr):
    '''年化收益率'''
    e = len(arr) / 252
    annualized_return = (arr[-1]**(1/e)-1) * 100
    return annualized_return


def Volatility(arr_return):
    '''年化波动率'''
    annualized_std = np.std(arr_return) * (np.sqrt(252)) * 100
    return annualized_std


def MaxDrawdown(arr):
    '''最大回撤率'''
    i = np.argmax((np.maximum.accumulate(arr) - arr) / np.maximum.accumulate(arr))#结束位置
    if i == 0:
        return 0
    j = np.argmax(arr[:i])  # 开始位置
    maxdrawdown = (arr[j] - arr[i]) / (arr[j]) * 100
    return maxdrawdown, j, i


def MaxDrawdown_Restore_Time(arr, df):
    '''最大回撤恢复时间'''
    def MDD_Restore(startidx, endidx, arr):       
        restore_time = 0
        restore_endidx = np.inf
        for t in range(endidx, len(arr)):
            if arr[t] >= arr[startidx]:
                restore_endidx = t
                break
            else:
                restore_time += 1
        restore_endidx = min(restore_endidx, len(arr)-1)
        return restore_time, restore_endidx

    for idx in df.index:
        df.loc[idx,'MDD_restore_time'] = MDD_Restore(df.loc[idx,'MDD_start_idx'],df.loc[idx,'MDD_end_idx'],
                                                     arr.loc[:,idx])[0]
    return df['MDD_restore_time']



df = pd.DataFrame(index=data.columns)

df['Annualized_Returns'] = data.apply(lambda x: Annualized_Returns(x))
df['Annualized_Volatility'] = Volatility(data_return)
df['Sharpe_ratio'] = df['Annualized_Returns'] / df['Annualized_Volatility']  
df['Max_Drawdown'] = data.apply(lambda x: MaxDrawdown(x)[0])    
df['MDD_start_idx'] = data.apply(lambda x: MaxDrawdown(x.values)[1])
df['MDD_end_idx'] = data.apply(lambda x: MaxDrawdown(x.values)[2])    
df['MDD_restore_time'] = MaxDrawdown_Restore_Time(data, df)   

df['MDD_start_time'] = df['MDD_start_idx'].map(lambda x: data.index[x])
df['MDD_end_time'] = df['MDD_end_idx'].map(lambda x: data.index[x])


df_out = df[['Annualized_Returns', 'Annualized_Volatility', 'Sharpe_ratio',
       'Max_Drawdown','MDD_start_time', 'MDD_end_time','MDD_restore_time']]
df_out.columns = ['年化收益率','年化波动率','夏普比率','最大回撤','最大回撤开始时间','最大回撤结束时间','最大回撤恢复所需时间']
df_out.to_excel('组合指标计算结果.xlsx')










    
    