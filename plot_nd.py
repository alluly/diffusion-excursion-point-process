import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

plt.style.use('seaborn-darkgrid')
sns.set(font_scale=2)

experiments = ['tanh10', 'cubic10', 'lv2', 'ou5']

for ex in experiments:

    with open('experiments/synthetic/{}-bb-log-10.pkl'.format(ex), 'rb') as f:
        data = np.stack(pickle.load(f))

    with open('experiments/synthetic/{}-ex-log-10.pkl'.format(ex), 'rb') as f:
        datae = np.stack(pickle.load(f))

    print('bb {}'.format(data.mean()))
    print('e {}'.format(datae.mean()))

    l2_e  = 'MSE\n(ex)'
    l2_bb = 'MSE\n(BB)'
    rel_e = 'Relative\n(ex)'
    rel_bb = 'Relative\n(BB)'

    #df = pd.DataFrame(np.concatenate((datae, data),1), columns=[r'$\|\cdot\|_2^2 (e_t)$',r'$\frac{\|\cdot\|_2^2}{\|\cdot\|_2^2} (e_t)$', r'$L^2 (BB_t)$', r'$\frac{L^2}{ \| L^2 \|} (BB_t)$'])
    df = pd.DataFrame(np.concatenate((datae, data),1), columns=[l2_e, rel_e, l2_bb, rel_bb])
    df = df[[l2_e, l2_bb, rel_e, rel_bb]]
    #df = df[['MSE (Excursion)', 'MSE (Bridge)', 'Normalized (Excursion)', 'Normalized (Bridge)']]
    #df = df[[r'$L^2 (e_t)$', r'$L^2 (BB_t)$', r'$\text{Relative} (e_t)$',r'$\text{Relative} (BB_t)$']]
    #df = df[[r'$\|\cdot\|_2^2 (e_t)$', r'$L^2 (BB)$', r'$\frac{\|\cdot\|_2^2}{\|\cdot\|_2^2} (e_t)$',r'$\frac{L^2}{ \| L^2 \|} (BB)$']]
    g = sns.boxplot(data=df, palette='pastel')
    plt.tight_layout()
    plt.savefig('figs/{}-bb_mse-log-10.pdf'.format(ex))
    plt.close('all')

