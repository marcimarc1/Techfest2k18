import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
import sklearn.mixture as mix
from flask import Flask
import json
app = Flask(__name__)

@app.route('/')
def predictWithCluster(timestamp):
    #load cluster models
    filename_cluster = 'clustermodel.sav'
    cluster_models = pickle.load(open(filename_cluster, 'rb'))
    #load regression model
    filename_reg= 'regmodel.sav'
    reg_models = pickle.load(open(filename_reg,'rb'))

    X = np.array(range(1,4))*timestamp

    '''
    Data order:
    contiform_effectivepower
    filler_PerformanceFactor
    filler_Temperature
    labeller_CurrentSpeed
    labeller_EffectivePower
    mixer_EffectivePower
    mixer_WaterFlow
    petview_CurrentSpeed
    '''
    l=['contiform_effectivepower',
    'filler_PerformanceFactor',
    'filler_Temperature',
    'labeller_CurrentSpeed',
    'labeller_EffectivePower',
    'mixer_EffectivePower',
    'mixer_WaterFlow',
    'petview_CurrentSpeed']

    #predict next variables
    Z={}
    for (reg,w) in zip(reg_models,l):
        Z[w] = reg.predict(X.reshape(-1,1))


    #cluster them
    cluster={}
    for (cl,w) in zip(cluster_models,l):
        cluster[w] =cl.predict(Z[w].reshape(-1,1))

    for i in l:
        Z[i] = Z[i].tolist()
        cluster[i] = cluster[i].tolist()
    return  json.dumps(Z), json.dumps(cluster), json.dumps(X.tolist())



Z,cluster,X = predictWithCluster(1)

print(Z)
print(cluster)
print(X)