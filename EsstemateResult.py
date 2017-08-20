# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 14:06:42 2017

@author: gramma
"""

import matplotlib.pylab as plt
import pandas as pd
import numpy as np

def Esstemate(Ypredict,Yfact,Vmin,Vmax):
    df1 = pd.DataFrame(Ypredict)
    df2=pd.DataFrame(Yfact)
    if (df1.shape!=df2.shape):
        raise ValueError('shape Y1 != shape Y2')
    (rowsCount,colCount)=df1.shape
    dV=(float)(Vmax-Vmin)
    for i in range(colCount):
        df1[colCount+i]=abs(df1[i]-df2[i])*dV/(df2[i]*dV+Vmin)
    df1[colCount+colCount]=df1[[x for x in range(colCount,2*colCount)]].mean(axis=1)
    df1[2*colCount+1]=df1[[x for x in range(colCount,2*colCount)]].max(axis=1)
    print(df1)
