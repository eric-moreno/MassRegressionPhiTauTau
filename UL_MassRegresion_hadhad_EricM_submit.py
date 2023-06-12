#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams.update({'font.size': 14})
import numpy as np
import math
import random
import setGPU

import h5py
import sklearn
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy as sc
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
import pandas as pd

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import os 
os.chdir('/home/submit/emoreno/emoreno/phitautau/MassRegression')

from preprocessor import load, load_flat


# In[2]:
print(tf.config.list_physical_devices('GPU'))


# In[35]:


from preprocessor import load, load_flat
X_flat,Xalt_flat,Xevt_flat,y_flat,feat_flat = load_flat('/data/submit/emoreno/phitautau/Apr21/FlatTauTau_user_noLep.z',maxevts=0);
X_htt2,Xalt_htt2,Xevt_htt2,y_htt2,feat_htt2 = load_flat('/data/submit/emoreno/phitautau/Apr21/GluGluHToTauTau_user_noLep.z'); #6k
#X_htt2,Xalt_htt2,Xevt_htt2,y_htt2,feat_htt2 = load_flat('/uscms_data/d3/keiran/CMSSW_10_2_11/src/PandaAnalysis/dazsle-tagger/evt/massReg_UL/Feb24/GluGluHToTauTau_user_noLep.z'); #100k
X_z,Xalt_z, Xevt_z, y_z, feat_z = load_flat('/data/submit/emoreno/phitautau/Apr21/tt-DYJetsToLL_noLep.z');
X_qcd,Xalt_qcd,Xevt_qcd,y_qcd,feat_qcd = load_flat('/data/submit/emoreno/phitautau/Apr21/QCD_noLep.z');

otherH = True
if otherH: 
    X_htt1,Xalt_htt1,Xevt_htt1,y_htt1,feat_htt1 = load_flat('/data/submit/emoreno/phitautau/Apr22/OtherHToTauTau_user_noLep.z'); #100k
    X_htt2 = np.concatenate((X_htt2, X_htt1))
    Xalt_htt2 = np.concatenate((Xalt_htt2, Xalt_htt1))
    Xevt_htt2 = np.concatenate((Xevt_htt2, Xevt_htt1))
    y_htt2 = np.concatenate((y_htt2, y_htt1))
    feat_htt2 = np.concatenate((feat_htt2, feat_htt1))


# In[40]:


X_htt = np.vstack([X_htt2])
Xalt_htt = np.vstack([Xalt_htt2])
Xevt_htt = np.vstack([Xevt_htt2])
y_htt = np.vstack([y_htt2])
feat_htt = np.vstack([feat_htt2])


# In[41]:


X_htt, Xalt_htt, Xevt_htt, y_htt = sklearn.utils.shuffle(X_htt, Xalt_htt, Xevt_htt, y_htt)
X_z, Xalt_z, Xevt_z, y_z = sklearn.utils.shuffle(X_z, Xalt_z, Xevt_z, y_z)
X_flat, Xalt_flat, Xevt_flat, y_flat = sklearn.utils.shuffle(X_flat, Xalt_flat, Xevt_flat, y_flat)
X_qcd, Xalt_qcd, Xevt_qcd, y_qcd, feat_qcd = sklearn.utils.shuffle(X_qcd, Xalt_qcd, Xevt_qcd, y_qcd, feat_qcd)


# In[42]:


# cuts on pT:
#flat_msd_cut = np.where((Xevt_flat[:,-4]>10)&(Xevt_flat[:,-4]<400)&(Xevt_flat[:,4]>40)&(Xevt_flat[:,-3]>300)&(Xevt_flat[:,-3]<400))
#htt_msd_cut = np.where((Xevt_htt[:,-4]>10)&(Xevt_htt[:,-4]<400)&(Xevt_htt[:,4]>40)&(Xevt_htt[:,-3]>300)&(Xevt_htt[:,-3]<400))
#z_msd_cut = np.where((Xevt_z[:,-4]>10)&(Xevt_z[:,-4]<400)&(y_z[:,-4]>5)&(Xevt_z[:,4]>40)&(Xevt_z[:,-3]>300)&(Xevt_z[:,-3]<400))
#qcd_msd_cut = np.where((Xevt_qcd[:,-4]>10)&(Xevt_qcd[:,-4]<400)&(Xevt_qcd[:,-3]>300)&(Xevt_qcd[:,-3]<400))

# no cut on pT (normal training method)
flat_msd_cut = np.where((Xevt_flat[:,-4]>10)&(Xevt_flat[:,-4]<400)&(Xevt_flat[:,4]>40))
htt_msd_cut = np.where((Xevt_htt[:,-4]>10)&(Xevt_htt[:,-4]<400)&(Xevt_htt[:,4]>40))
z_msd_cut = np.where((Xevt_z[:,-4]>10)&(Xevt_z[:,-4]<400)&(y_z[:,-4]>5)&(Xevt_z[:,4]>40))
#qcd_msd_cut = np.where((Xevt_qcd[:,-4]>10)&(Xevt_qcd[:,-4]<400))
qcd_msd_cut = np.where((Xevt_qcd[:,-4]>10)&(Xevt_qcd[:,-4]<400)&(feat_qcd[:,15]>150)) # MET>150 Cut

X_flat,Xalt_flat,Xevt_flat,y_flat,feat_flat = X_flat[flat_msd_cut], Xalt_flat[flat_msd_cut], Xevt_flat[flat_msd_cut], y_flat[flat_msd_cut], feat_flat[flat_msd_cut]
X_htt,Xalt_htt,Xevt_htt,y_htt,feat_htt = X_htt[htt_msd_cut], Xalt_htt[htt_msd_cut], Xevt_htt[htt_msd_cut], y_htt[htt_msd_cut], feat_htt[htt_msd_cut]
X_z,Xalt_z,Xevt_z,y_z,feat_z = X_z[z_msd_cut], Xalt_z[z_msd_cut], Xevt_z[z_msd_cut], y_z[z_msd_cut], feat_z[z_msd_cut]
X_qcd,Xalt_qcd,Xevt_qcd,y_qcd,feat_qcd = X_qcd[qcd_msd_cut], Xalt_qcd[qcd_msd_cut], Xevt_qcd[qcd_msd_cut], y_qcd[qcd_msd_cut], feat_qcd[qcd_msd_cut]

pffeatures = ["PF_pt","PF_eta","PF_phi","PF_q","PF_dz","PF_dzerr","PF_d0","PF_d0err","PF_pup","PF_pupnolep","PF_id","PF_trk","PF_vtx"]
altfeatures = [ "sv_dlen", "sv_dlenSig", "sv_dxy", "sv_dxySig", "sv_chi2", "sv_pAngle", "sv_x", "sv_y", "sv_z", "sv_pt", "sv_mass", "sv_eta", "sv_phi"]
evtfeatures = ["MET_covXX","MET_covXY","MET_covYY","MET_phi","MET_pt","MET_significance","PuppiMET_pt","PuppiMET_phi","fj_msd","fj_pt","fj_eta","fj_phi"]

for iv,v in enumerate(pffeatures): 
    plt.figure()
    plt.hist(X_flat[:,:,iv], bins = 30, histtype='step', density=True)
    plt.title(pffeatures[iv])
    plt.savefig("/home/submit/emoreno/public_html/work/phitautau/MassRegression/plots/PF #" + str(iv) + ".jpg")

for iv,v in enumerate(altfeatures): 
    plt.figure()
    plt.hist(Xalt_flat[:,:,iv], bins = 30, histtype='step', density=True)
    plt.title(altfeatures[iv])
    plt.savefig("/home/submit/emoreno/public_html/work/phitautau/MassRegression/plots/SV #" + str(iv) + ".jpg")

for iv,v in enumerate(evtfeatures): 
    plt.figure()
    plt.hist(Xevt_flat[:,iv], bins = 30, histtype='step', density=True)
    plt.title(evtfeatures[iv])
    plt.savefig("/home/submit/emoreno/public_html/work/phitautau/MassRegression/plots/EVT #" + str(iv) + ".jpg")
    plt.close

plt.figure()
plt.show()
plt.hist(Xevt_flat[:,-4],bins=np.arange(0,400,7), density=True, alpha=0.5, label='flat');
plt.hist(Xevt_z[:,-4],bins=np.arange(0,400,7), density=True, alpha=0.5, label='z');
plt.hist(Xevt_htt[:,-4],bins=np.arange(0,400,7), density=True, alpha=0.5, label='htt');
plt.hist(Xevt_qcd[:,-4],bins=np.arange(0,400,7), density=True, alpha=0.5, label='qcd');
plt.legend()
plt.title('sdmass (postcut)')
                       
plt.figure()
plt.show()
plt.hist(Xevt_flat[:,4],bins=np.arange(0,400,7), density=True, alpha=0.5, label='flat');
plt.hist(Xevt_z[:,4],bins=np.arange(0,400,7), density=True, alpha=0.5, label='z');
plt.hist(Xevt_htt[:,4],bins=np.arange(0,400,7), density=True, alpha=0.5, label='htt');
plt.hist(Xevt_qcd[:,4],bins=np.arange(0,400,7), density=True, alpha=0.5, label='qcd');
plt.legend()
plt.title('what variable is this?')


# In[43]:


prefix = "figs/"

####### Plot pT distribution ########
plt.figure()
plt.hist(Xevt_flat[:,-3],bins=np.arange(0,500,1));

summary = f'''mean:{np.mean(Xevt_flat[:,-3])}
std:{np.std(Xevt_flat[:,-3])}'''

anchored_text = AnchoredText(summary, loc=1)
#anchored_text2 = AnchoredText(f"std:{np.std(Xevt_flat[:,-4])}", loc=4)

ax = plt.gca()
ax.add_artist(anchored_text)
#ax.add_artist(anchored_text2)
plt.title('pT distribution Flat')
plt.savefig(prefix+'flat_pT.pdf')

plt.figure()
plt.hist(Xevt_htt[:,-3],bins=np.arange(0,500,1));

summary = f'''mean:{np.mean(Xevt_htt[:,-3])}
std:{np.std(Xevt_htt[:,-3])}'''

anchored_text = AnchoredText(summary, loc=1)
#anchored_text2 = AnchoredText(f"std:{np.std(Xevt_flat[:,-4])}", loc=4)

ax = plt.gca()
ax.add_artist(anchored_text)
#ax.add_artist(anchored_text2)
plt.title('pT distribution htt')
plt.savefig(prefix+'htt_pT.pdf')

plt.figure()
plt.hist(Xevt_z[:,-3],bins=np.arange(0,500,1));

summary = f'''mean:{np.mean(Xevt_z[:,-3])}
std:{np.std(Xevt_z[:,-3])}'''

anchored_text = AnchoredText(summary, loc=1)
#anchored_text2 = AnchoredText(f"std:{np.std(Xevt_flat[:,-4])}", loc=4)

ax = plt.gca()
ax.add_artist(anchored_text)
#ax.add_artist(anchored_text2)
plt.title('pT distribution z')
plt.savefig(prefix+'z_pT.pdf')
plt.show()


plt.figure()
plt.show()
plt.hist(Xevt_flat[:,-3],bins=np.arange(0,1500,10), density=True, alpha=0.5, label='flat');
plt.hist(Xevt_z[:,-3],bins=np.arange(0,1500,10), density=True, alpha=0.5, label='z');
plt.hist(Xevt_htt[:,-3],bins=np.arange(0,1500,10), density=True, alpha=0.5, label='htt');
plt.hist(Xevt_qcd[:,-3],bins=np.arange(0,1500,10), density=True, alpha=0.5, label='qcd');
plt.legend()
plt.title('pT')

plt.figure()
plt.hist(y_flat[:,-4],bins=np.arange(0,300,5), density=True, label='flat');
plt.hist(y_z[:,-4],bins=np.arange(0,300,5), density=True, label='z');
plt.hist(y_htt[:,-4],bins=np.arange(0,300,5), density=True, label='htt');
plt.hist(y_qcd[:,-4],bins=np.arange(0,300,5), density=True, label='qcd');
plt.legend()
plt.title('Ground truth mass')


plt.figure()
plt.show()
plt.hist(Xevt_flat[:,-4],bins=np.arange(0,400,10), density=True, alpha=0.5, label='flat');
plt.hist(Xevt_z[:,-4],bins=np.arange(0,400,10), density=True, alpha=0.5, label='z');
plt.hist(Xevt_htt[:,-4],bins=np.arange(0,400,10), density=True, alpha=0.5, label='htt');
plt.hist(Xevt_qcd[:,-4],bins=np.arange(0, 400,10), density=True, alpha=0.5, label='qcd');
plt.legend()
plt.title('sdmass (precut)')

plt.figure()
plt.show()
plt.hist(X_flat[:,:,10].flatten(),bins=np.arange(0,40,1),  density=True, alpha=0.5, label='flat');
plt.hist(X_z[:,:,10].flatten(),bins=np.arange(0,40,1), density=True, alpha=0.5, label='z');
plt.hist(X_htt[:,:10].flatten(), bins=np.arange(0,40,1),density=True, alpha=0.5, label='htt');
plt.legend()
plt.title('PID')
plt.show()


# In[ ]:




# In[44]:


###### Network Parameters ########

particlesConsidered = 30
entriesPerParticle = 13

svConsidered = 5
entriesPerSV = 13

eventDataLength = 12

numberOfEpochs = 200
batchSize = 2048

# Loss Function
from tensorflow.python.ops import math_ops
from scipy import stats

def mean_sqrt_error(y_true, y_pred):
    diff = math_ops.abs((y_true - y_pred) / K.maximum(tf.sqrt(y_true), K.epsilon()))
    return diff[...,0]+lamb*diff[...,1]

MLP_model = False

num_H = [1000,5000,10000,25000,50000, 90000]
num_Z = [1000,5000,10000,25000,50000, 90000]
lamb1 = [0.01]
lamb2 = [0.1]
lamb3 = [1]
lamb4 = [10]

#wps = np.array(np.meshgrid(num_H, num_Z, lamb1)).T.reshape(-1, 3)

# [(num Htt in training, num Z in training, lambda)]
wps = [(1000,1000,0.01), (5000,5000,0.01), (25000,25000,0.01), (50000,50000,0.01) ]
TEST_SIZE = 120000
HOW_MANY_FLAT_TO_TRAINING = 900000


# In[45]:


class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, x_test, y_test):
        self.model  = model
        self.x_test = x_test
        self.y_test = y_test
        
    def on_epoch_end(self, epoch, logs={}):
        #print(type(self.validation_data))
        y_pred = self.model.predict(self.x_test)
        print('prediction: {} at epoch: {}'.format(y_pred, epoch))
        print('prediction Truth: {} at epoch: {}'.format(self.y_test, epoch))


# In[46]:


print(X_htt.shape, X_z.shape, X_flat.shape)


# In[47]:


print(Xalt_htt.shape, Xalt_z.shape, Xalt_flat.shape)


# In[48]:


print(Xevt_htt.shape, Xevt_z.shape, Xevt_flat.shape)


# In[ ]:





# In[ ]:





# In[49]:


##### Network Training ########
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

for HOW_MANY_H_TO_TRAINING, HOW_MANY_Z_TO_TRAINING, lamb in wps:
            HOW_MANY_H_TO_TRAINING = int(HOW_MANY_H_TO_TRAINING)
            HOW_MANY_Z_TO_TRAINING = int(HOW_MANY_Z_TO_TRAINING)
            
            modelName=f"UL_H{HOW_MANY_H_TO_TRAINING}_Z{HOW_MANY_Z_TO_TRAINING}_FLAT{HOW_MANY_FLAT_TO_TRAINING}_Lambda{lamb}_hadhad"
            if MLP_model: 
                modelName=f"UL_MLP_H{HOW_MANY_H_TO_TRAINING}_Z{HOW_MANY_Z_TO_TRAINING}_FLAT{HOW_MANY_FLAT_TO_TRAINING}_Lambda{lamb}_hadhad"
            
            #modelName=f"Mass_and_Pt_IN_H{HOW_MANY_H_TO_TRAINING}_Z{HOW_MANY_Z_TO_TRAINING}_Lambda{lamb}_FLAT500k_genPtCut400_msd5_met50"
            
            X_train = np.vstack((X_htt[:HOW_MANY_H_TO_TRAINING], X_z[:HOW_MANY_Z_TO_TRAINING], X_flat[:HOW_MANY_FLAT_TO_TRAINING]))
            Xalt_train = np.vstack((Xalt_htt[:HOW_MANY_H_TO_TRAINING], Xalt_z[:HOW_MANY_Z_TO_TRAINING], Xalt_flat[:HOW_MANY_FLAT_TO_TRAINING]))
            Xevt_train = np.vstack((Xevt_htt[:HOW_MANY_H_TO_TRAINING], Xevt_z[:HOW_MANY_Z_TO_TRAINING], Xevt_flat[:HOW_MANY_FLAT_TO_TRAINING]))
            y_train = np.concatenate((y_htt[:HOW_MANY_H_TO_TRAINING,0:2], y_z[:HOW_MANY_Z_TO_TRAINING,0:2], y_flat[:HOW_MANY_FLAT_TO_TRAINING,0:2]))
            #plt.hist(y_train[:,0])
            #plt.hist(y_train[:,1])
            #plt.show()
            X_test_htt = X_htt[HOW_MANY_H_TO_TRAINING :HOW_MANY_H_TO_TRAINING +TEST_SIZE]
            X_test_z   = X_z[HOW_MANY_Z_TO_TRAINING :HOW_MANY_Z_TO_TRAINING +TEST_SIZE]

            Xalt_test_htt = Xalt_htt[HOW_MANY_H_TO_TRAINING :HOW_MANY_H_TO_TRAINING +TEST_SIZE]
            Xalt_test_z   = Xalt_z[HOW_MANY_Z_TO_TRAINING :HOW_MANY_Z_TO_TRAINING +TEST_SIZE]

            Xevt_test_htt = Xevt_htt[HOW_MANY_H_TO_TRAINING :HOW_MANY_H_TO_TRAINING +TEST_SIZE]
            Xevt_test_z = Xevt_z[HOW_MANY_Z_TO_TRAINING :HOW_MANY_Z_TO_TRAINING +TEST_SIZE]

            y_test_htt = y_htt[HOW_MANY_H_TO_TRAINING :HOW_MANY_H_TO_TRAINING +TEST_SIZE,0:2]
            y_test_z  = y_z[HOW_MANY_Z_TO_TRAINING :HOW_MANY_Z_TO_TRAINING +TEST_SIZE,0:2]

            X_test_flat   = X_flat[HOW_MANY_FLAT_TO_TRAINING :HOW_MANY_FLAT_TO_TRAINING +TEST_SIZE]
            Xalt_test_flat   = Xalt_flat[HOW_MANY_FLAT_TO_TRAINING :HOW_MANY_FLAT_TO_TRAINING +TEST_SIZE]
            Xevt_test_flat = Xevt_flat[HOW_MANY_FLAT_TO_TRAINING :HOW_MANY_FLAT_TO_TRAINING +TEST_SIZE]
            y_test_flat  = y_flat[HOW_MANY_FLAT_TO_TRAINING :HOW_MANY_FLAT_TO_TRAINING +TEST_SIZE,0:2]

            X_train, Xalt_train, Xevt_train, y_train = sklearn.utils.shuffle(X_train, Xalt_train, Xevt_train, y_train)
            X_train, X_val, Xalt_train, Xalt_val, Xevt_train, Xevt_val, y_train, y_val = train_test_split(X_train, Xalt_train, Xevt_train, y_train, test_size=0.2, random_state=42)
            
            ###### Network Architecture ######

            ## receiving matrix

            RR = []
            for i in range(particlesConsidered):
                row = []
                for j in range(particlesConsidered * (particlesConsidered - 1)):
                    if j in range(i * (particlesConsidered - 1), (i + 1) * (particlesConsidered - 1)):
                        row.append(1.0)
                    else:
                        row.append(0.0)
                RR.append(row)
            RR = np.array(RR)
            RR = np.float32(RR)
            RRT = np.transpose(RR)

            ## sending matrix

            RST = []
            for i in range(particlesConsidered):
                for j in range(particlesConsidered):
                    row = []
                    for k in range(particlesConsidered):
                        if k == j:
                            row.append(1.0)
                        else:
                            row.append(0.0)
                    RST.append(row)
            rowsToRemove = []
            for i in range(particlesConsidered):
                rowsToRemove.append(i * (particlesConsidered + 1))
            RST = np.array(RST)
            RST = np.float32(RST)
            RST = np.delete(RST, rowsToRemove, 0)
            RS = np.transpose(RST)

            ## recieving matrix for the bipartite particle and secondary vertex graph

            RK = []
            for i in range(particlesConsidered):
                row = []
                for j in range(particlesConsidered * svConsidered):
                    if j in range(i * svConsidered, (i + 1) * svConsidered):
                        row.append(1.0)
                    else:
                        row.append(0.0)
                RK.append(row)
            RK = np.array(RK)
            RK = np.float32(RK)
            RKT = np.transpose(RK)

            ## defines the sending matrix for the bipartite particle and secondary vertex graph


            RV = []
            for i in range(svConsidered):
                row = []
                for j in range(particlesConsidered * svConsidered):
                    if j % svConsidered == i:
                        row.append(1.0)
                    else:
                        row.append(0.0)
                RV.append(row)
            RV = np.array(RV)
            RV = np.float32(RV)
            RVT = np.transpose(RV)

            inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")

            XdotRR = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RR, axes=[[2], [0]]),
                                                        perm=(0, 2, 1)), name="XdotRR")(inputParticle)
            XdotRS = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RS, axes=[[2], [0]]),
                                                        perm=(0, 2, 1)), name="XdotRS")(inputParticle)
            Bpp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bpp")([XdotRR, XdotRS])

            convOneParticle = Conv1D(60, kernel_size=1, activation="relu", name="convOneParticle")(Bpp)
            convTwoParticle = Conv1D(30, kernel_size=1, activation="relu", name="convTwoParticle")(convOneParticle)
            convThreeParticle = Conv1D(20, kernel_size=1, activation="relu", name="convThreeParticle")(convTwoParticle)

            Epp = BatchNormalization(momentum=0.6, name="Epp")(convThreeParticle)

            # Secondary vertex data interaction NN
            inputSV = Input(shape=(svConsidered, entriesPerSV), name="inputSV")

            XdotRK = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RK, axes=[[2], [0]]),
                                                        perm=(0, 2, 1)), name="XdotRK")(inputParticle)
            YdotRV = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RV, axes=[[2], [0]]),
                                                        perm=(0, 2, 1)), name="YdotRV")(inputSV)
            Bvp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bvp")([XdotRK, YdotRV])

            convOneSV = Conv1D(60, kernel_size=1, activation="relu", name="convOneSV")(Bvp)
            convTwoSV = Conv1D(30, kernel_size=1, activation="relu", name="convTwoSV")(convOneSV)
            convThreeSV = Conv1D(20, kernel_size=1, activation="relu", name="convThreeSV")(convTwoSV)

            Evp = BatchNormalization(momentum=0.6, name="Evp")(convThreeSV)


            # Event Level Info

            inputEvent = Input(shape=(eventDataLength, ), name="inputEvent")


            # Combined prediction NN
            EppBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RRT, axes=[[2], [0]]),
                                                        perm=(0, 2, 1)), name="EppBar")(Epp)
            EvpBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RKT, axes=[[2], [0]]),
                                                        perm=(0, 2, 1)), name="EvpBar")(Evp)
            C = Lambda(lambda listOfTensors: tf.concat((listOfTensors[0], listOfTensors[1], listOfTensors[2]), axis=2), name="C")(
                [inputParticle, EppBar, EvpBar])

            convPredictOne = Conv1D(60, kernel_size=1, activation="relu", name="convPredictOne")(C)
            convPredictTwo = Conv1D(30, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)

            O = Conv1D(24, kernel_size=1, activation="relu", name="O")(convPredictTwo)

            # Calculate output
            OBar = Lambda(lambda tensor: K.sum(tensor, axis=1), name="OBar")(O)

            Concatted = Concatenate()([OBar, inputEvent])

            denseEndOne = Dense(50, activation="relu", name="denseEndOne")(Concatted)
            normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
            denseEndTwo = Dense(20, activation="relu", name="denseEndTwo")(normEndOne)
            denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
            output = Dense(2,name="output")(denseEndThree)


            Concatted = Concatenate()([inputEvent])
            denseEndOne = Dense(50, activation="relu", name="denseEndOne")(Concatted)
            normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
            denseEndTwo = Dense(20, activation="relu", name="denseEndTwo")(normEndOne)
            denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
            outputMLP = Dense(2,name="output")(denseEndThree)


            if MLP_model:
                model = Model(inputs=[inputParticle, inputSV, inputEvent], outputs=[outputMLP]) #MLP Model
            else: 
                model = Model(inputs=[inputParticle, inputSV, inputEvent], outputs=[output])
            opt = Adam(learning_rate=0.001)
            model.compile(optimizer=opt, loss=mean_sqrt_error, metrics=[mean_sqrt_error], run_eagerly=True)
            modelCallbacks = [EarlyStopping(patience=7),ModelCheckpoint(filepath="./weights/"+modelName+".h5", save_weights_only=True, save_best_only=True)]
            PredictionCallback(model, [X_val, Xalt_val, Xevt_val], y_val )
            history = model.fit([X_train, Xalt_train, Xevt_train], y_train, epochs=numberOfEpochs, batch_size=batchSize,
                                callbacks=modelCallbacks,
                                validation_data=([X_val, Xalt_val, Xevt_val], y_val))


            model.save("./saved_model/"+modelName)
            model.load_weights("./weights/"+modelName+".h5")
            predictions_htt = model.predict([X_test_htt, Xalt_test_htt, Xevt_test_htt])
            predictions_z = model.predict([X_test_z, Xalt_test_z, Xevt_test_z])
            
            
            massH = predictions_htt[:,0]
            H_rec = (massH.flatten()-y_test_htt[:,0])/y_test_htt[:,0]
            
            massZ = predictions_z[:,0]
            Z_rec = (massZ.flatten()-y_test_z[:,0])/y_test_z[:,0]

            predictions_flat = model.predict([X_test_flat, Xalt_test_flat, Xevt_test_flat])
            massFlat = predictions_flat[:,0]
            Flat_rec = (massFlat.flatten()-y_test_flat[:,0])/y_test_flat[:,0]            
            
            n, b, patches = plt.hist(H_rec,bins=np.arange(-0.75,0.75,0.01),alpha=.9,histtype='step',density=True,color='r',label='H');
            H_mode = b[np.argmax(n)]

            n, b, patches = plt.hist(Z_rec,bins=np.arange(-0.75,0.75,0.01),alpha=.9,histtype='step',density=True,color='g',label='Z');
            Z_mode = b[np.argmax(n)]
            n, b, patches = plt.hist(Flat_rec,bins=np.arange(-0.75,0.75,0.01),alpha=.9,histtype='step',density=True,color='b',label='Flat');
            Flat_mode = b[np.argmax(n)]
            plt.xlim([-1.2, 1.2])
            print(Z_mode)
            plt.text(0.75,1.3,f'Flat Mean: {np.mean(Flat_rec):.3f}, Mode:{Flat_mode:.3f}',fontsize=12,transform=ax.transAxes)
            plt.text(0.75,1.4,f'H Mean: {np.mean(H_rec):.3f}, Mode:{H_mode:.3f}',fontsize=12,transform=ax.transAxes)
            plt.text(0.75,1.5,f'Z Mean: {np.mean(Z_rec):.3f}, Mode:{Z_mode:.3f}',fontsize=12,transform=ax.transAxes)

            plt.xlabel(r'$(m_{reg}-m_{gen})/m_{gen}$',fontsize=15)
            plt.ylabel(r'$density$',fontsize=15)
            plt.legend(loc='upper right')
            plt.savefig(prefix+f'Zmode:{Z_mode:.3f}-'+modelName+'-mass-response.png')
            plt.show()
            K.clear_session() 


# In[50]:

##### Network Evaluation ########

from preprocessor import load, load_flat

#loading in only ggH events
#X_htt2,Xalt_htt2,Xevt_htt2,y_htt2,feat_htt2 = load_flat('/uscms_data/d3/keiran/CMSSW_10_2_11/src/PandaAnalysis/dazsle-tagger/evt/massReg_UL_hadhad/Apr21/GluGluHToTauTau_user_noLep.z'); #6k
X_htt2,Xalt_htt2,Xevt_htt2,y_htt2,feat_htt2 = load_flat('/data/submit/emoreno/phitautau/Apr21/GluGluHToTauTau_user_noLep.z'); #6k
X_htt = np.vstack([X_htt2])
Xalt_htt = np.vstack([Xalt_htt2])
Xevt_htt = np.vstack([Xevt_htt2])
y_htt = np.vstack([y_htt2])
feat_htt = np.vstack([feat_htt2])
X_htt, Xalt_htt, Xevt_htt, y_htt = sklearn.utils.shuffle(X_htt, Xalt_htt, Xevt_htt, y_htt)
htt_msd_cut = np.where((Xevt_htt[:,-4]>10)&(Xevt_htt[:,-4]<400)&(Xevt_htt[:,4]>40))
X_htt,Xalt_htt,Xevt_htt,y_htt,feat_htt = X_htt[htt_msd_cut], Xalt_htt[htt_msd_cut], Xevt_htt[htt_msd_cut], y_htt[htt_msd_cut], feat_htt[htt_msd_cut]

X_w,Xalt_w,Xevt_w,y_w,feat_w = load_flat('/data/submit/emoreno/phitautau/Apr21/WJetsToLNu_noLep.z', maxevts=100000); 
X_w = np.vstack([X_w])
Xalt_w = np.vstack([Xalt_w])
Xevt_w = np.vstack([Xevt_w])
y_w = np.vstack([y_w])
feat_w = np.vstack([feat_w])
X_w, Xalt_w, Xevt_w, y_w = sklearn.utils.shuffle(X_w, Xalt_w, Xevt_w, y_w)
w_msd_cut = np.where((Xevt_w[:,-4]>10)&(Xevt_w[:,-4]<400)&(Xevt_w[:,4]>40))
X_w,Xalt_w,Xevt_w,y_w,feat_w = X_w[w_msd_cut], Xalt_w[w_msd_cut], Xevt_w[w_msd_cut], y_w[w_msd_cut], feat_w[w_msd_cut]


for HOW_MANY_H_TO_TRAINING, HOW_MANY_Z_TO_TRAINING, lamb in wps:
            
    HOW_MANY_H_TO_TRAINING = int(HOW_MANY_H_TO_TRAINING)
    HOW_MANY_Z_TO_TRAINING = int(HOW_MANY_Z_TO_TRAINING)
    
    modelName=f"UL_H{HOW_MANY_H_TO_TRAINING}_Z{HOW_MANY_Z_TO_TRAINING}_FLAT{HOW_MANY_FLAT_TO_TRAINING}_Lambda{lamb}_hadhad"
    if MLP_model: 
        modelName=f"UL_MLP_H{HOW_MANY_H_TO_TRAINING}_Z{HOW_MANY_Z_TO_TRAINING}_FLAT{HOW_MANY_FLAT_TO_TRAINING}_Lambda{lamb}_hadhad"
    #modelName=f"Mass_and_Pt_IN_H{HOW_MANY_H_TO_TRAINING}_Z{HOW_MANY_Z_TO_TRAINING}_Lambda{lamb}_FLAT500k_genPtCut400_msd5_met50"
    #HOW_MANY_H_TO_TRAINING_temp =  HOW_MANY_H_TO_TRAINING
    #HOW_MANY_Z_TO_TRAINING_temp = HOW_MANY_Z_TO_TRAINING
    #HOW_MANY_FLAT_TO_TRAINING_temp = HOW_MANY_FLAT_TO_TRAINING
    
    #HOW_MANY_H_TO_TRAINING = 0 
    #HOW_MANY_Z_TO_TRAINING = 0
    #HOW_MANY_FLAT_TO_TRAINING = 10000
    
    X_train = np.vstack((X_htt[:HOW_MANY_H_TO_TRAINING], X_z[:HOW_MANY_Z_TO_TRAINING], X_flat[:HOW_MANY_FLAT_TO_TRAINING]))
    Xalt_train = np.vstack((Xalt_htt[:HOW_MANY_H_TO_TRAINING], Xalt_z[:HOW_MANY_Z_TO_TRAINING], Xalt_flat[:HOW_MANY_FLAT_TO_TRAINING]))
    Xevt_train = np.vstack((Xevt_htt[:HOW_MANY_H_TO_TRAINING], Xevt_z[:HOW_MANY_Z_TO_TRAINING], Xevt_flat[:HOW_MANY_FLAT_TO_TRAINING]))
    y_train = np.concatenate((y_htt[:HOW_MANY_H_TO_TRAINING,0:2], y_z[:HOW_MANY_Z_TO_TRAINING,0:2], y_flat[:HOW_MANY_FLAT_TO_TRAINING,0:2]))
    plt.hist(y_train[:,0])
    #plt.hist(y_train[:,1])
    plt.show()
    X_test_htt = X_htt[:TEST_SIZE]
    X_test_z   = X_z[HOW_MANY_Z_TO_TRAINING :HOW_MANY_Z_TO_TRAINING +TEST_SIZE]

    Xalt_test_htt = Xalt_htt[:TEST_SIZE]
    Xalt_test_z   = Xalt_z[HOW_MANY_Z_TO_TRAINING :HOW_MANY_Z_TO_TRAINING +TEST_SIZE]

    Xevt_test_htt = Xevt_htt[:TEST_SIZE]
    Xevt_test_z = Xevt_z[HOW_MANY_Z_TO_TRAINING :HOW_MANY_Z_TO_TRAINING +TEST_SIZE]

    y_test_htt = y_htt[:TEST_SIZE,0:2]
    y_test_z  = y_z[HOW_MANY_Z_TO_TRAINING :HOW_MANY_Z_TO_TRAINING +TEST_SIZE,0:2]

    X_test_flat   = X_flat[HOW_MANY_FLAT_TO_TRAINING :HOW_MANY_FLAT_TO_TRAINING +TEST_SIZE]
    Xalt_test_flat   = Xalt_flat[HOW_MANY_FLAT_TO_TRAINING :HOW_MANY_FLAT_TO_TRAINING +TEST_SIZE]
    Xevt_test_flat = Xevt_flat[HOW_MANY_FLAT_TO_TRAINING :HOW_MANY_FLAT_TO_TRAINING +TEST_SIZE]
    y_test_flat  = y_flat[HOW_MANY_FLAT_TO_TRAINING :HOW_MANY_FLAT_TO_TRAINING +TEST_SIZE,0:2]
    
    X_train, Xalt_train, Xevt_train, y_train = sklearn.utils.shuffle(X_train, Xalt_train, Xevt_train, y_train)
    X_train, X_val, Xalt_train, Xalt_val, Xevt_train, Xevt_val, y_train, y_val = train_test_split(X_train, Xalt_train, Xevt_train, y_train, test_size=1, random_state=42)
    
    #HOW_MANY_H_TO_TRAINING =  HOW_MANY_H_TO_TRAINING_temp
    #HOW_MANY_Z_TO_TRAINING = HOW_MANY_Z_TO_TRAINING_temp  
    #HOW_MANY_FLAT_TO_TRAINING = HOW_MANY_FLAT_TO_TRAINING_temp
    
    ###### Network Architecture ######

    ## receiving matrix

    RR = []
    for i in range(particlesConsidered):
        row = []
        for j in range(particlesConsidered * (particlesConsidered - 1)):
            if j in range(i * (particlesConsidered - 1), (i + 1) * (particlesConsidered - 1)):
                row.append(1.0)
            else:
                row.append(0.0)
        RR.append(row)
    RR = np.array(RR)
    RR = np.float32(RR)
    RRT = np.transpose(RR)

    ## sending matrix

    RST = []
    for i in range(particlesConsidered):
        for j in range(particlesConsidered):
            row = []
            for k in range(particlesConsidered):
                if k == j:
                    row.append(1.0)
                else:
                    row.append(0.0)
            RST.append(row)
    rowsToRemove = []
    for i in range(particlesConsidered):
        rowsToRemove.append(i * (particlesConsidered + 1))
    RST = np.array(RST)
    RST = np.float32(RST)
    RST = np.delete(RST, rowsToRemove, 0)
    RS = np.transpose(RST)

    ## recieving matrix for the bipartite particle and secondary vertex graph

    RK = []
    for i in range(particlesConsidered):
        row = []
        for j in range(particlesConsidered * svConsidered):
            if j in range(i * svConsidered, (i + 1) * svConsidered):
                row.append(1.0)
            else:
                row.append(0.0)
        RK.append(row)
    RK = np.array(RK)
    RK = np.float32(RK)
    RKT = np.transpose(RK)

    ## defines the sending matrix for the bipartite particle and secondary vertex graph


    RV = []
    for i in range(svConsidered):
        row = []
        for j in range(particlesConsidered * svConsidered):
            if j % svConsidered == i:
                row.append(1.0)
            else:
                row.append(0.0)
        RV.append(row)
    RV = np.array(RV)
    RV = np.float32(RV)
    RVT = np.transpose(RV)

    inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")

    XdotRR = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RR, axes=[[2], [0]]),
                                                perm=(0, 2, 1)), name="XdotRR")(inputParticle)
    XdotRS = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RS, axes=[[2], [0]]),
                                                perm=(0, 2, 1)), name="XdotRS")(inputParticle)
    Bpp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bpp")([XdotRR, XdotRS])

    convOneParticle = Conv1D(60, kernel_size=1, activation="relu", name="convOneParticle")(Bpp)
    convTwoParticle = Conv1D(30, kernel_size=1, activation="relu", name="convTwoParticle")(convOneParticle)
    convThreeParticle = Conv1D(20, kernel_size=1, activation="relu", name="convThreeParticle")(convTwoParticle)

    Epp = BatchNormalization(momentum=0.6, name="Epp")(convThreeParticle)

    # Secondary vertex data interaction NN
    inputSV = Input(shape=(svConsidered, entriesPerSV), name="inputSV")

    XdotRK = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RK, axes=[[2], [0]]),
                                                perm=(0, 2, 1)), name="XdotRK")(inputParticle)
    YdotRV = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RV, axes=[[2], [0]]),
                                                perm=(0, 2, 1)), name="YdotRV")(inputSV)
    Bvp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bvp")([XdotRK, YdotRV])

    convOneSV = Conv1D(60, kernel_size=1, activation="relu", name="convOneSV")(Bvp)
    convTwoSV = Conv1D(30, kernel_size=1, activation="relu", name="convTwoSV")(convOneSV)
    convThreeSV = Conv1D(20, kernel_size=1, activation="relu", name="convThreeSV")(convTwoSV)

    Evp = BatchNormalization(momentum=0.6, name="Evp")(convThreeSV)


    # Event Level Info

    inputEvent = Input(shape=(eventDataLength, ), name="inputEvent")


    # Combined prediction NN
    EppBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RRT, axes=[[2], [0]]),
                                                perm=(0, 2, 1)), name="EppBar")(Epp)
    EvpBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RKT, axes=[[2], [0]]),
                                                perm=(0, 2, 1)), name="EvpBar")(Evp)
    C = Lambda(lambda listOfTensors: tf.concat((listOfTensors[0], listOfTensors[1], listOfTensors[2]), axis=2), name="C")(
        [inputParticle, EppBar, EvpBar])

    convPredictOne = Conv1D(60, kernel_size=1, activation="relu", name="convPredictOne")(C)
    convPredictTwo = Conv1D(30, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)

    O = Conv1D(24, kernel_size=1, activation="relu", name="O")(convPredictTwo)

    # Calculate output
    OBar = Lambda(lambda tensor: K.sum(tensor, axis=1), name="OBar")(O)

    Concatted = Concatenate()([OBar, inputEvent])

    denseEndOne = Dense(50, activation="relu", name="denseEndOne")(Concatted)
    normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
    denseEndTwo = Dense(20, activation="relu", name="denseEndTwo")(normEndOne)
    denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
    output = Dense(2,name="output")(denseEndThree)


    Concatted = Concatenate()([inputEvent])
    denseEndOne = Dense(50, activation="relu", name="denseEndOne")(Concatted)
    normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
    denseEndTwo = Dense(20, activation="relu", name="denseEndTwo")(normEndOne)
    denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
    outputMLP = Dense(2,name="output")(denseEndThree)

    
    if MLP_model:
        model = Model(inputs=[inputParticle, inputSV, inputEvent], outputs=[outputMLP]) #MLP Model
    else: 
        model = Model(inputs=[inputParticle, inputSV, inputEvent], outputs=[output])
    
    
        
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=mean_sqrt_error, metrics=[mean_sqrt_error], run_eagerly=True)
    modelCallbacks = [EarlyStopping(patience=10),ModelCheckpoint(filepath="./weights/"+modelName+".h5", save_weights_only=True, save_best_only=True)]
    PredictionCallback(model, [X_val, Xalt_val, Xevt_val], y_val )

    #model.load_weights("./weights/"+modelName+".h5")
    model.load_weights("weights/NEW_gpu_UL_NoFeatSel_H20000_Z25000_FLAT500000_Lambda0.01_hadhad.h5")
    #model.save("./saved_model/"+modelName)

    
    HOW_MANY_H_TO_TRAINING = 25000
    
    predictions_htt = model.predict([X_test_htt, Xalt_test_htt, Xevt_test_htt])
    predictions_z = model.predict([X_test_z, Xalt_test_z, Xevt_test_z])
    predictions_w = model.predict([X_w, Xalt_w, Xevt_w])
    
    plt.figure()
    _, bins_z, patches_z = plt.hist(predictions_z[:,0],bins=np.arange(50,200,0.5),linewidth=1.5,density=True,facecolor="None");
    _, bins_h, patches_h = plt.hist(predictions_htt[:,0],bins=np.arange(50,200,0.5),linewidth=1.5,density=True,facecolor="None");
    plt.hist(predictions_z[:,0],bins=np.arange(50,200,1),linewidth=2,density=True,histtype="step",color="C0",label="Z");
    plt.hist(predictions_htt[:,0],bins=np.arange(50,200,1),linewidth=2,density=True,histtype="step",color="C1",label="Higgs");
    plt.axvline(x=np.quantile(predictions_htt[:,0],0.1),color='C1',linewidth=2)
    for i in range(len(bins_z)-1):
        #print(bins_h[i])
        if bins_z[i] > np.quantile(predictions_htt[:,0],0.1):
            plt.setp(patches_z[i], facecolor="C0",alpha=0.3)


    for i in range(len(bins_h)-1):
        #print(bins_h[i])
        if bins_h[i] > np.quantile(predictions_htt[:,0],0.1):
            plt.setp(patches_h[i], facecolor="C1",alpha=0.3)

    plt.legend(loc='upper left')    
    plt.text(np.quantile(predictions_htt[:,0],0.1),0.015,'90% of Higgs to the Right',ha='center', va='center',rotation='vertical', backgroundcolor='None')
    score_complement=stats.percentileofscore(predictions_z[:,0],np.quantile(predictions_htt[:,0],0.1))/100
    score = 1.-score_complement
    summary = r'$P(m_{Z,reg} > cutoff)=$'f'{score:.3f}''\n'r'$1 / \sqrt{P} = $'f'{1/np.sqrt(score):.3f}'
    anchored_text = AnchoredText(summary, loc=1)

    ax = plt.gca()
    ax.add_artist(anchored_text)
    plt.title(f'{HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed, Lambda={lamb}')

    plt.savefig(prefix+f'summaryscore-{1/np.sqrt(score):.4f}'+modelName+f'.png')
    plt.show()
    
    plt.figure()
    count_z, bins_z, patches_z = plt.hist(predictions_z[:,0],bins=np.arange(50,210,10),linewidth=1.5,density=True,facecolor="None");
    count_h, bins_h, patches_h = plt.hist(predictions_htt[:,0],bins=np.arange(50,210,10),linewidth=1.5,density=True,facecolor="None");
    plt.hist(predictions_z[:,0],bins=np.arange(50,210,10),linewidth=2,density=True,histtype="step",color="C0",label="Z");
    plt.hist(predictions_htt[:,0],bins=np.arange(50,210,10),linewidth=2,density=True,histtype="step",color="C1",label="Higgs");
    #plt.axvline(x=110,color='r',linewidth=2,alpha=0.4)
    #plt.axvline(x=140,color='r',linewidth=2,alpha=0.4)
    S = 0.
    B = 0.
    for i in range(len(bins_z)-1):
        #print(bins_h[i])
        if bins_z[i] in [110,120,130]:
            plt.setp(patches_z[i], facecolor="C0",alpha=0.2)
            plt.setp(patches_h[i], facecolor="C1",alpha=0.2)
            S += count_h[i]
            B += count_z[i]

    significance = S/math.sqrt(B)
    summary = r'$\frac{P(m_{H,reg} \in [110,140])}{\sqrt{P(m_{Z,reg} \in [110,140])}}=$'f'{significance:.3f}'
    anchored_text = AnchoredText(summary, loc=1,prop=dict(fontsize=15))
    plt.legend(loc='upper left')    

    ax = plt.gca()
    ax.add_artist(anchored_text)
    plt.title(f'{HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed, Lambda={lamb}')

    plt.savefig(prefix+f'significance-{np.sqrt(significance):.4f}'+modelName+f'.png')

    plt.show()
    
    plt.figure()
    #predictions_qcd = model.predict([X_qcd, Xalt_qcd, Xevt_qcd])
    predictions_qcd = model([X_qcd, Xalt_qcd, Xevt_qcd])
    #predicitons_qcd = model.predict([X_test_htt, Xalt_test_htt, Xevt_test_htt]) #just for short inference if QCD plots dont matter
    mass_qcd = predictions_qcd[:,0]

    #H_rec = (mass_qcd.flatten()-y_test_htt[:,0])/y_test_htt[:,0]

    plt.hist(mass_qcd,bins=np.arange(0,300,5),color='C0',label='Reconstructed Mass QCD',alpha=0.2,density=True);
    #plt.hist(y_qcd[:,0],bins=np.arange(0,300,5),color='C1',label='gen Mass QCD',alpha=0.2,density=True)
    plt.hist(predictions_htt[:,0],bins=np.arange(0,300,5),color='C1',label='Reconstructed Mass Htt',alpha=0.2,density=True);
    plt.hist(predictions_w[:,0],bins=np.arange(0,300,5),color='C2',label='Reconstructed Mass W',alpha=0.2,density=True);
    plt.hist(predictions_z[:,0],bins=np.arange(0,300,5),color='C3',label='Reconstructed Mass Z',alpha=0.2,density=True);
    #plt.hist(y_htt[:,0],bins=np.arange(0,300,5),color='C3',label='gen Mass Htt',alpha=0.2,density=True);
    plt.axvline(x=125,color='r')
    #plt.text(110,500,'m=125',rotation=90,fontsize=15)
    #plt.title(f'QCD, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    plt.title(f'QCD, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    plt.legend()
    plt.xlabel('M (GeV)')
    plt.ylabel('Density')
    #std:{np.std(mass_qcd)}

    #anchored_text = AnchoredText(summary, loc=1)
    #ax = plt.gca()
    #ax.add_artist(anchored_text)
    plt.savefig(prefix+modelName+f'-QCD-hadhad-mass.png')
    plt.show()
    
    plt.figure()
    plt.hist(predictions_qcd[:,1],bins=np.arange(0,1000,10),color='C0',label='Reconstructed pT QCD',alpha=0.2,density=True);
    #plt.hist(y_qcd[:,1],bins=np.arange(0,1000,10),color='C1',label='gen pT QCD',alpha=0.2,density=True)
    plt.hist(predictions_htt[:,1],bins=np.arange(0,1000,10),color='C1',label='Reconstructed pT Htt',alpha=0.2,density=True);
    plt.hist(predictions_w[:,1],bins=np.arange(0,1000,10),color='C2',label='Reconstructed pT W',alpha=0.2,density=True);
    plt.hist(predictions_z[:,1],bins=np.arange(0,1000,10),color='C3',label='Reconstructed pT Z',alpha=0.2,density=True);
    #plt.hist(y_htt[:,1],bins=np.arange(0,1000,10),color='C3',label='gen pT Htt',alpha=0.2,density=True);
    #plt.title(f'QCD, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    plt.title(f'QCD, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    plt.xlabel('pT (GeV)')
    plt.legend()
    plt.ylabel('Density')
    plt.savefig(prefix+modelName+f'-QCD-hadhad-pT.png')
    plt.show()

    '''
    # Plot SEP old model vs New models
    plt.figure()
    plt.hist(predictions_qcd[:,0],bins=np.arange(0,300,5),color='C0',label='Reconstructed Mass QCD NEW',alpha=0.2,density=True);
    plt.hist(SEPpredictions_qcd[:,0],bins=np.arange(0,300,5),color='C0',label='Reconstructed Mass QCD OLD',alpha=0.2,density=True);
    plt.axvline(x=125,color='r')
    plt.title(f'QCD, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    lt.legend()
    plt.xlabel('M (GeV)')
    plt.ylabel('Density')
    plt.savefig(prefix+modelName+f'-QCD-hadhad-Mass_comparison.png')
    
    plt.figure()
    plt.hist(predictions_htt[:,0],bins=np.arange(0,300,5),color='C0',label='Reconstructed Mass Htt NEW',alpha=0.2,density=True);
    plt.hist(SEPpredictions_htt[:,0],bins=np.arange(0,300,5),color='C0',label='Reconstructed Mass Htt OLD',alpha=0.2,density=True);
    plt.axvline(x=125,color='r')
    plt.title(f'Htt, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    lt.legend()
    plt.xlabel('M (GeV)')
    plt.ylabel('Density')
    plt.savefig(prefix+modelName+f'-Htt-hadhad-Mass_comparison.png')

    plt.figure()
    plt.hist(predictions_w[:,0],bins=np.arange(0,300,5),color='C0',label='Reconstructed Mass W NEW',alpha=0.2,density=True);
    plt.hist(SEPpredictions_w[:,0],bins=np.arange(0,300,5),color='C0',label='Reconstructed Mass W OLD',alpha=0.2,density=True);
    plt.axvline(x=125,color='r')
    plt.title(f'W, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    lt.legend()
    plt.xlabel('M (GeV)')
    plt.ylabel('Density')
    plt.savefig(prefix+modelName+f'-W-hadhad-Mass_comparison.png')

    plt.figure()
    plt.hist(predictions_z[:,0],bins=np.arange(0,300,5),color='C0',label='Reconstructed Mass Z NEW',alpha=0.2,density=True);
    plt.hist(SEPpredictions_z[:,0],bins=np.arange(0,300,5),color='C0',label='Reconstructed Mass Z OLD',alpha=0.2,density=True);
    plt.axvline(x=125,color='r')
    plt.title(f'Z, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    lt.legend()
    plt.xlabel('M (GeV)')
    plt.ylabel('Density')
    plt.savefig(prefix+modelName+f'-Z-hadhad-Mass_comparison.png')

    plt.figure()
    plt.hist(predictions_qcd[:,1],bins=np.arange(0,300,5),color='C0',label='Reconstructed pT QCD NEW',alpha=0.2,density=True);
    plt.hist(SEPpredictions_qcd[:,1],bins=np.arange(0,300,5),color='C0',label='Reconstructed pT QCD OLD',alpha=0.2,density=True);
    plt.axvline(x=125,color='r')
    plt.title(f'QCD, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    lt.legend()
    plt.xlabel('pT (GeV)')
    plt.ylabel('Density')
    plt.savefig(prefix+modelName+f'-QCD-hadhad-pT_comparison.png')
    
    plt.figure()
    plt.hist(predictions_htt[:,1],bins=np.arange(0,300,5),color='C0',label='Reconstructed pT Htt NEW',alpha=0.2,density=True);
    plt.hist(SEPpredictions_htt[:,1],bins=np.arange(0,300,5),color='C0',label='Reconstructed pT Htt OLD',alpha=0.2,density=True);
    plt.axvline(x=125,color='r')
    plt.title(f'Htt, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    lt.legend()
    plt.xlabel('pT (GeV)')
    plt.ylabel('Density')
    plt.savefig(prefix+modelName+f'-Htt-hadhad-pT_comparison.png')

    plt.figure()
    plt.hist(predictions_w[:,1],bins=np.arange(0,300,5),color='C0',label='Reconstructed pT W NEW',alpha=0.2,density=True);
    plt.hist(SEPpredictions_w[:,1],bins=np.arange(0,300,5),color='C0',label='Reconstructed pT W OLD',alpha=0.2,density=True);
    plt.axvline(x=125,color='r')
    plt.title(f'W, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    lt.legend()
    plt.xlabel('pT (GeV)')
    plt.ylabel('Density')
    plt.savefig(prefix+modelName+f'-W-hadhad-pT_comparison.png')

    plt.figure()
    plt.hist(predictions_z[:,1],bins=np.arange(0,300,5),color='C0',label='Reconstructed pT Z NEW',alpha=0.2,density=True);
    plt.hist(SEPpredictions_z[:,1],bins=np.arange(0,300,5),color='C0',label='Reconstructed pT Z OLD',alpha=0.2,density=True);
    plt.axvline(x=125,color='r')
    plt.title(f'Z, MET>150, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    lt.legend()
    plt.xlabel('pT (GeV)')
    plt.ylabel('Density')
    plt.savefig(prefix+modelName+f'-Z-hadhad-pT_comparison.png')
    '''

    plt.figure()
    pt_response = (predictions_htt[:,1]-y_test_htt[:,1])/y_test_htt[:,1]

    plt.hist(pt_response,bins=np.arange(-0.5,0.5,.02),color='r',label=r'$p_{T}$ Response',alpha=0.2, density=True);
    #plt.hist(y_htt[:,1],bins=np.arange(0,1000,10),color='b',label='gen Pt',alpha=0.2)
    plt.title(f'Htt, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed')
    plt.xlabel(r'$(p_{T,reg}-p_{T,gen})/p_{T,gen}$')
    plt.ylabel('Density')
    plt.legend(loc='upper left')

    summary = f'''mean: {np.mean(pt_response):.3f}\nstd:   {np.std(pt_response):.3f}'''

    anchored_text = AnchoredText(summary, loc=1)
    ax = plt.gca()
    ax.add_artist(anchored_text)


    plt.savefig(prefix+modelName+f'-Higgs-hadhad-pT-Response-summary.png')
    plt.show()
    plt.figure()
    massH = predictions_htt[:,0]
    H_rec = (massH.flatten()-y_test_htt[:,0])/y_test_htt[:,0]
    
    massZ = predictions_z[:,0]
    Z_rec = (massZ.flatten()-y_test_z[:,0])/y_test_z[:,0]

    #predictions_flat = model.predict([X_test_flat, Xalt_test_flat, Xevt_test_flat])
    predictions_flat = model([X_test_flat, Xalt_test_flat, Xevt_test_flat])
    massFlat = predictions_flat[:,0]
    Flat_rec = (massFlat.flatten()-y_test_flat[:,0])/y_test_flat[:,0]            
    
    massW = predictions_w[:,0]
    W_rec = (massW.flatten()-y_w[:,0])/(y_w[:,0]+0.01)
    
    plt.hist(H_rec,bins=np.arange(-0.75,0.75,0.02),alpha=.9,histtype='step',density=True,color='r',label='H');
    plt.hist(Z_rec,bins=np.arange(-0.75,0.75,0.02),alpha=.9,histtype='step',density=True,color='g',label='Z');
    plt.hist(Flat_rec,bins=np.arange(-0.75,0.75,0.02),alpha=.9,histtype='step',density=True,color='b',label='Flat');
    plt.xlabel(r'$(m_{reg}-m_{gen})/m_{gen}$',fontsize=15)
    plt.ylabel(r'$density$',fontsize=15)
    plt.legend(loc='upper right')
    plt.title(f'Htt, {HOW_MANY_H_TO_TRAINING}H and {HOW_MANY_Z_TO_TRAINING}Z mixed, pT>200')
    plt.savefig(prefix+modelName+'-mass-response.png')
    plt.show()
    
    #K.clear_session() 


# In[ ]:




