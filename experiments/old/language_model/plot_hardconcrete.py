import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

limit_a = -0.1
limit_b = 1.1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sample_hardconcrete(x, m, beta=1.0):
    shape = (x.size, m)
    noise = np.random.uniform(0, 1, shape)
    s = sigmoid((np.log(noise) - np.log(1-noise) + x.reshape(-1, 1))/beta)
    s_bar = s * (limit_b - limit_a) + limit_a
    y = np.clip(s_bar, 0.0, 1.0)
    return y

def sample_hardsigmoid(x, l=-0.1, r=1.1):
    y = sigmoid(x*0.8)
    y = y * (r-l) + l
    y = np.clip(y, 0.0, 1.0)
    return y

N = 2000
M = 8000
eps = 1e-8

x = [ 20*(i/2000.0)-10 for i in range(2001) ]
x = np.asarray(x)

y_hc = sample_hardconcrete(x, M, beta=1.0)
y_hc_2 = sample_hardconcrete(x, M, beta=1.0)
y_sg = sample_hardsigmoid(x)
percent_zeros = (y_hc < eps).sum(axis=1) / (M+0.0)
percent_ones = (y_hc > 1-eps).sum(axis=1) / (M+0.0)

diff = y_hc.mean(axis=1)-y_sg
print (diff.min(), diff.max())

sns.set(style='whitegrid')
f, axs = plt.subplots(figsize=(8, 4.5), ncols=1, nrows=1)
sns.set(palette='Set2')

palette = sns.color_palette()
hc = sns.tsplot(data=y_hc[:,:200].T, time=x, ax=axs, ci=99, linewidth=0.7, color=palette[1])
#hc_2 = sns.tsplot(data=y_hc_2[:,:200].T, time=x, ax=axs, ci=99, linewidth=0.7, color=palette[2])
sg = sns.tsplot(data=y_sg.T, time=x, ax=axs, ci='sd', linewidth=1.2, color=palette[2])
plt.legend([hc, sg], labels=['E[hard concrete]', 'hardsigmoid(0.8x)'])
#plt.legend([hc, hc_2, sg], labels=['hard concrete (temperature=1)', 'hard concreate (temperature=1.5)', 'hard sigmoid'])
#p_0 = sns.tsplot(data=percent_zeros, time=x, ax=axs, linewidth=0.5, color=palette[0])
#p_1 = sns.tsplot(data=percent_ones, time=x, ax=axs, linewidth=0.5, color=palette[4])
#plt.legend([hc, p_0, p_1], labels=['hard concrete', 'percentage of 0', 'percentage of 1'])

#print (x[1300])
#sns.distplot(y_hc[1300])
plt.show()

