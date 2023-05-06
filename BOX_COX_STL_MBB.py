import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from pandas import read_csv
import pandas as pd
from statsmodels.tsa.seasonal import STL
from arch.bootstrap import MovingBlockBootstrap
from numpy.random import RandomState
from numpy.random import standard_normal

'''BOX-COX变换部分'''
ts = np.loadtxt(open('/capacity_37_1.csv'), delimiter=",",
                skiprows=0)  # CSV文件转化为数组(数组或者矩阵存储为csv文件:numpy.savetxt('new.csv', ts, delimiter = ',')
# plt.subplot(211)  # 211代表2*1的矩阵图放第一个
plt.plot(ts, 'green')
y, lambda1 = boxcox(ts)  # 正向Box-Cox变换
print('lambda1=', lambda1)
# plt.plot(y)
# plt.show()

'''STL分解部分'''
stl = STL(y, period=50, robust=True)
res_robust = stl.fit()  # 估计季节、趋势和残差成分
# print(res_robust.trend)
# print(res_robust.seasonal)
# print(res_robust.resid)
data = pd.Series(data=res_robust.resid)
data.to_excel('C:\myPython\my_others\data\stl_resid.xlsx', index=False, header=False)     # 保存残差数据
# fig = res_robust.plot()
# plt.show()
data = pd.read_excel('C:\myPython\my_others\data\stl_resid.xlsx', 'Sheet1', index_col=0)
data.to_csv('C:\myPython\my_others\data\stl_resid.csv', encoding='utf-8')  # 将excel转换为.csv文件

'''MBB部分'''
stl_resid = np.loadtxt(open('C:\myPython\my_others\data\stl_resid.csv'), delimiter=",",
                skiprows=0)  # CSV文件转化为数组(数组或者矩阵存储为csv文件:numpy.savetxt('new.csv', ts, delimiter = ',')
bs = MovingBlockBootstrap(14, stl_resid=stl_resid)  # block_size为14
bs_stl_resid = np.array([])
for data in bs.bootstrap(100):  # 引导程序复制的数量为100
    bs_stl_resid = bs.stl_resid
# plt.plot(stl_resid,'red')
# plt.plot(bs_stl_resid)
# plt.show()
# print(type(bs_stl_resid))
# np.savetxt("C:\myPython\my_others\data\ bs_stl_resid.txt", bs_stl_resid)  # 储存np.array数组

# '''STL合成部分'''
stl_synthesize = res_robust.trend + res_robust.seasonal + bs_stl_resid
# plt.plot(stl_synthesize)
# plt.show()
np.savetxt("C:\myPython\my_others\data\ stl_synthesize.txt", stl_synthesize)  # 储存np.array数组

'''BOX-COX逆变换部分'''  # 有问题：逆变换后会出现数据丢失
new_series = inv_boxcox(stl_synthesize, lambda1)  # 逆向Box-Cox变换
print(type(new_series))
np.savetxt("C:\myPython\my_others\data\ new_series.txt", new_series)  # 储存np.array数组
data = pd.Series(data=new_series)
data.to_csv('C:\myPython\my_others\data\ new_series.csv', index=False, header=False)     # 保存残差数据
plt.plot(new_series, 'red')
plt.show()


