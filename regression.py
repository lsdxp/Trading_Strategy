import numpy as np
import scikits.statsmodels.api as sm
import import_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import scipy.stats as stats
import matplotlib.mlab as mlab
import pandas as pandas
import statsmodels.graphics.tsaplots as stat_graph
tsa = sm.tsa

len=1170
prices=import_data.get_gold(len)
prices2=import_data.get_silver(len)
prices3=import_data.get_pt(len)
date=import_data.get_date(len)

mylist=[a/b for a,b in zip(prices,prices2)]
endog=np.asarray(mylist)
exog = np.column_stack([np.asarray(prices),
                        np.asarray(prices2)])
exog = sm.add_constant(exog, prepend=True)
res1 = sm.OLS(endog, exog).fit()

# acf, ci, Q, pvalue = tsa.acf(res1.resid, nlags=20,
#                              confint=95, qstat=True,
# unbiased=True)
# acfx = np.linspace(0,19,20)
#
# plt.plot(acfx,acf[:20])
# plt.show()


# res2=np.cumsum(res1.resid[:20])
# plt.plot(res2)
# plt.show()

#print res1.summary()

def unit_root_test(endog):
    print tsa.adfuller(endog, regression="ct")[:2]#adf-statistic and a p-value ,
    print tsa.adfuller(np.diff(np.log(endog)), regression="ct")[:2]

def linear_regression(prices,prices2):
    prices=[[a] for a in prices]
    prices2=[[a] for a in prices2]
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(prices, prices2)

    #
    # plt.scatter(prices, prices2,  color='black')
    # plt.plot(prices, regr.predict(prices), color='blue',
    #          linewidth=3)
    # plt.xticks(())
    # plt.yticks(())

    d=regr.predict(prices)-prices2

    plt.figure(1)
    plt.hist(d, normed=True)
    plt.xlim((min(d), max(d)))

    mean = np.mean(d)
    variance = np.var(d)
    sigma = np.sqrt(variance)
    x = np.linspace(min(d), max(d),100)
    plt.plot(x,mlab.normpdf(x,mean,sigma))
    plt.show()

def multi_vector_regression(p,p2,p3,date):

    len=1170
    len2=1170-len
    test_p=p[len:]
    test_p2=p2[len:]
    test_p3=p3[len:]
    test_date=date[len:]

    test_data = pandas.DataFrame({'Gold' : test_p,
    'SLV' : test_p2,
    'PPLT':test_p3
    })

    import statsmodels.tsa.api
    from statsmodels.tsa.base.datetools import dates_from_str
    from pandas.tseries.offsets import *
    fig=np.zeros((len2,3))

    train_p=p[:len]
    train_p2=p2[:len]
    train_p3=p3[:len]
    train_date=date[:len]

    data = pandas.DataFrame({'Gold' : train_p,
    'SLV' : train_p2,
    'PPLT':train_p3
    })
    data.to_csv('/Users/shengdongliu/Trading_Strategy/output_data/metal/toR2.csv')
    train_date = dates_from_str(train_date)
    data.index = pandas.DatetimeIndex(train_date)

    model = statsmodels.tsa.api.VAR(data)
    results = model.fit(1)

    lag_order = results.k_ar
    print data.values[-lag_order:]
    fig=results.forecast(data.values[-lag_order:],len2)

    # test_date = dates_from_str(test_date)
    # test_data.index = pandas.DatetimeIndex(test_date)

    # make a VAR model

    # for x in range(len+1,len+len2):
    #     temp=[[p[x],p2[x],p3[x]]]
    #     fig[x-len]=results.forecast(temp,1)

    plt.plot(test_data)
    plt.plot(fig)
    #results.plot()
    plt.show()
    #print test_data
    #print results.summary()
    # print model.select_order(15)
    # lag_order = results.k_ar
    # print results.forecast(data.values[-lag_order:], 5)


multi_vector_regression(prices,prices2,prices3,date)
