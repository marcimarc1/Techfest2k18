import itertools
import numpy as np
import pandas as pd
import sklearn.mixture as mix
import sklearn.linear_model as model
from scipy import linalg
import matplotlib.pyplot as plt
import pickle
import sklearn.mixture as mix
#Reading in the data as Pandas df
df_conti_ep = pd.read_csv('contiform_effectivepower.csv')
df_filler_pf = pd.read_csv('filler_PerformanceFactor.csv')
df_filler_t = pd.read_csv('filler_Temperature.csv')
df_labeller_cs = pd.read_csv('labeller_CurrentSpeed.csv')
df_labeller_ep = pd.read_csv('labeller_EffectivePower.csv')
df_mixer_ep = pd.read_csv('mixer_EffectivePower.csv')
df_mixer_wf = pd.read_csv('mixer_WaterFlow.csv')
df_pet_cs = pd.read_csv('petview_CurrentSpeed.csv')

# #Change index
#
# df_conti_ep.set_index('_time')
# df_filler_pf.set_index('_time')
# df_filler_t.set_index('_time')
# df_labeller_cs.set_index('_time')
# df_labeller_ep.set_index('_time')
# df_mixer_ep.set_index('_time')
# df_mixer_wf.set_index('_time')
# df_pet_cs.set_index('_time')
#
# #Mapping the time
# z = np.size(df_conti_ep)
# X =np.linspace(0,z/3)
# frames1 = [df_conti_ep,df_filler_pf,df_filler_t,df_labeller_cs,df_labeller_ep,df_mixer_ep,
#            df_mixer_wf,df_pet_cs]
#
#
# Y_primer = pd.concat(frames1)
#Getting the outputs


# frames = [df_conti_ep['EffectivePower'], df_filler_pf['PerformanceFactor'], df_filler_t['Temperature'],
#              df_labeller_cs['CurrentSpeed'],df_labeller_ep['EffectivePower'],
#              df_mixer_ep['EffectivePower'], df_mixer_wf['WaterFlow'], df_pet_cs['CurrentSpeed'] ]
# Y_prime= pd.concat(frames)


Y = [df_conti_ep['EffectivePower'].values[:], df_filler_pf['PerformanceFactor'].values[:], df_filler_t['Temperature'].values[:],
             df_labeller_cs['CurrentSpeed'].values[:],df_labeller_ep['EffectivePower'].values[:],
             df_mixer_ep['EffectivePower'].values[:], df_mixer_wf['WaterFlow'].values[:], df_pet_cs['CurrentSpeed'].values[:] ]

models = []

for Data in Y:
    Reg = model.LinearRegression()
    y=np.array(range(len(Data)))
    models.append(Reg.fit(Data.reshape(-1,1),y.T))


cluster_data =[]
for model in models:
    cluster_data.append(model.predict(y.T))

cluster_data = np.array(cluster_data)



assumed_clusters = 8
GMM = []
big = []
aic = []
different_cluster = []
for i in range(1,assumed_clusters):
    GMM.append(mix.GaussianMixture(n_components=i, covariance_type='full').fit(cluster_data))
    big.append(GMM[-1].bic(cluster_data))
    aic.append(GMM[-1].aic(cluster_data))

plt.plot(range(1,assumed_clusters),big)
plt.plot(range(1,assumed_clusters),aic)
plt.show()
