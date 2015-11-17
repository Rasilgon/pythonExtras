# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:28:45 2015

@author: trashtos
"""
# ----------------------------------------------------------------------------#
# Modules
# ----------------------------------------------------------------------------#
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# ----------------------------------------------------------------------------#
# Define metrics functions
# ----------------------------------------------------------------------------#

def enhancedConfusionMatrix(actuals, predictions):
    cm = confusion_matrix(actuals, predictions)
    #get total values
    truePositives = cm.diagonal()
    falsePositives = np.tril(cm, 0)#above diagonal
    falseNegatives = np.triu(cm, 0) #below diagonal
    # create list to hold data
    classesPrecision = np.zeros(len(cm[0]), dtype=np.float)
    classesAccuracy = np.zeros(len(cm[0]), dtype=np.float)
    for i in range(len(cm[0])):
        mask1D= np.ones(cm[0].shape,dtype=bool)
        mask1D[i]= 0
        truePos = cm[i,i]
        falsePos =np.sum(cm[:,i][mask1D])#below i
        falseNeg =np.sum(cm[i,:][mask1D])#same row than i 
        if truePos ==0:
            classPrecision = 0
            classAccuracy = 0
        else:
            classPrecision =  (float(truePos) / (truePos + falsePos))*100
            classAccuracy = (float(truePos) / (falseNeg + truePos))*100 #becuase is groudn truth
        #append values
        classesPrecision[i] = classPrecision
        classesAccuracy[i] =classAccuracy
    ####
    accuracy = (float(np.sum(truePositives)) /
                    (np.sum(truePositives)+ np.sum(falseNegatives)))*100
    precision  = (float(np.sum(truePositives)) /
                    (np.sum(truePositives)+ np.sum(falsePositives)))*100
    #####
    return (cm, accuracy,  precision,  classesPrecision, classesAccuracy)


def kappa(cm):
    
    N =  np.sum(cm.sum(axis=1)) + np.sum( cm.sum(axis=0))
    kappa = ( N*np.sum(cm.diagonal()) - np.sum(cm.sum(axis=1) *  cm.sum(axis=0))) / (N*N -  np.sum(cm.sum(axis=1) *  cm.sum(axis=0)))
     
    return(kappa)    
    
    
   
# ----------------------------------------------------------------------------#
# Define parameters
# ----------------------------------------------------------------------------#
np.random.seed(2)
numberSamples = 10
trainsplit = int(0.4 * numberSamples)
testSplit = int(1* numberSamples - trainsplit )
replicates = 10

# ----------------------------------------------------------------------------#
# Prepare data
# ----------------------------------------------------------------------------#
# Read in the table wioth attributes
fila = "/home/trashtos/masterarbeit/appendix/DataI/Tables/ok7clumps8080_14_check_clean_max_def.csv"
data = pd.read_csv(fila, sep=',',  header='infer', prefix ="NO")
#
data.loc[(data.Class == 23), ['Class']] = 1
data.loc[(data.Class == 1) & (data.CHMMax >= 2), ['Class']] = np.nan
#
data.loc[(data.Class == 3)  & (data.CHMMax >= 20), ['Class'] ] = 2#can select more classes
data.loc[(data.Class == 4)  & (data.CHMMax >= 6), ['Class'] ] = 3
    # elminate if too low
data.loc[(data.Class == 2)  & (data.CHMMax < 20), ['Class'] ] = np.nan
data.loc[(data.Class == 3)  & (data.CHMMax < 6), ['Class'] ] = np.nan#can select more classes
data.loc[(data.Class == 4)  & (data.CHMMax < 2), ['Class'] ] = np.nan
# 
#DW
data.loc[(data.Class == 5) & (data.CHMMax >= 3), ['Class']] = np.nan
data.loc[(data.Class == 6), ['Class']] = np.nan
data.loc[(data.Class == 7), ['Class'] ] = np.nan
data.loc[(data.Class == 8), ['Class']] = np.nan
data.loc[(data.Class == 9) & (data.CHMMax < 3), ['Class']] = np.nan
#dec
    #promote to next
data.loc[(data.Class == 11) & (data.CHMMax >= 20), ['Class']] = 10
data.loc[(data.Class == 12) & (data.CHMMax >= 6), ['Class']] = 11
    #elimnate if too  low
data.loc[(data.Class == 10) & (data.CHMMax < 20), ['Class']] =np.nan
data.loc[(data.Class == 11) & (data.CHMMax < 6), ['Class']] = np.nan
data.loc[(data.Class == 12) & (data.CHMMax < 2), ['Class']] = np.nan
# meadows
data.loc[(data.Class == 15), ['Class'] ] = 14
data.loc[(data.Class == 16), ['Class'] ] = 14
data.loc[(data.Class == 14) & (data.CHMMax >= 2), ['Class']] =np.nan
data.loc[(data.Class == 15) & (data.CHMMax >= 2), ['Class']] = np.nan
data.loc[(data.Class == 16) & (data.CHMMax >= 2), ['Class']] = np.nan
#mixed: check that is not there
data.loc[(data.Class == 17), ['Class']] = np.nan
data.loc[(data.Class == 18), ['Class']] = np.nan
data.loc[(data.Class == 19), ['Class']] = np.nan
#scrub pine 24
data.loc[(data.Class == 24) & (data.CHMMax < 2), ['Class']] = np.nan
data.loc[(data.Class == 22), ['Class']] = np.nan
data.loc[(data.Class == 25), ['Class']] = np.nan
data.loc[(data.Class == 26), ['Class']] = np.nan
data.loc[(data.Class == 13),  ['Class']] = np.nan
data.loc[(data.Class == 20),  ['Class']] = np.nan
###
cleanNA = data.dropna(axis=0, how='any', thresh=None, subset=None,
                      inplace=False).copy()

# ----------------------------------------------------------------------------#
# Importances
# ----------------------------------------------------------------------------#

# start with stuff
variables = list(cleanNA.columns[2:])
importances = np.zeros( (replicates,len(variables)) )

grouped = cleanNA.groupby('Class')

grouped.apply(lambda gb: np.random.choice(gb.ix,2))

for i in range(replicates):
    # start rf
    rf = RandomForestClassifier(n_estimators=1000, criterion='gini',
                                max_depth=None, min_samples_split=3,
                                min_samples_leaf=1,max_features=3,
                                bootstrap=True, oob_score=True, n_jobs=16,
                                random_state=2, verbose=2 ) 
    # Subset 100 first
    indicesTable =list(itertools.chain.from_iterable([np.random.choice(b.index.values, numberSamples, replace=False)   for a, b in grouped ]  ))
    table = cleanNA.ix[indicesTable]
    # Get ramdom selection of indices
    x = np.asarray(table.ix[:,2:],dtype=np.float32)
    response = np.asarray(table.ix[:,1].copy(),dtype=np.uint32)
    
    Counter(response)
    
#xtest = xtest.drop('CanopyCoverAvg', axis =1)
    sss = StratifiedShuffleSplit(response, 1, test_size=0.40, train_size=0.60,
                                 random_state=2)
    # do the stuff
    for test_index, train_index in sss:
        rf.fit(x[train_index], response[train_index])
        importances[i, :] = rf.feature_importances_
       # impVar = pd.DataFrame({'Variable':variables,'importances':importances})
      #  impVar.to_csv('~/Data/Importances/importance_' + str(i) + '.txt', index=False)

finalImportances = np.zeros(len(variables))
for i in range(len(variables)):
    finalImportances[i] = np.mean(importances[:,i])

importanceMean = pd.DataFrame({'Variable':variables,'importances':finalImportances})
#importanceMean.to_csv('~/Data/MeanImportance.txt', index=False, sep=',') 
  

# Now read data of importances and do the averate, then short them out
# importanceMean
rankedVariables = list (importanceMean.sort_index(by=['importances'], ascending=[False]).ix[:,0])


# ----------------------------------------------------------------------------#
# Classification
# ----------------------------------------------------------------------------#
# Start over with random forest
# create numpy array to hold values

for i in range(replicates):
    # start rf
    rf = RandomForestClassifier(n_estimators=1000, criterion='gini',
                                max_depth=None, min_samples_split=3,
                                min_samples_leaf=1,max_features=3,
                                bootstrap=True, oob_score=True, n_jobs=8,
                                random_state=2, verbose=2 ) 
        
    # Subset 100 first
    indicesTable =list(itertools.chain.from_iterable([np.random.choice(b.index.values, numberSamples, replace=False)   for a, b in grouped ]  ))
    table = cleanNA.ix[indicesTable]
    # Get ramdom selection of indices
    x = np.asarray(table.ix[:,2:],dtype=np.float32)
    response = np.asarray(table.ix[:,1].copy(),dtype=np.uint32)
    # split 
    sss = StratifiedShuffleSplit(response, 1, test_size=0.40, train_size=0.60,
                                 random_state=2)
    # do the stuff
    for test_index, train_index in sss:
        #loop over varibales regrarding to their importance
            for j in range(2, len(rankedVariables)):
                variablesSubset = rankedVariables[:j+1]
                x = np.asarray(table.loc[:,variablesSubset],dtype=np.float32)
                rf.fit(x[train_index], response[train_index])
                y_pred = rf.predict(x[test_index])
                y_true =  response[test_index]
                # get metrics
                cm, accuracy,  precision,  classesPrecision, classesAccuracy = enhancedConfusionMatrix(response[test_index], y_pred)
                classficationAcc = metrics.accuracy_score(response[test_index], y_pred)   
                score = rf.score(x[test_index], response[test_index] )
                classficationAcc = metrics.accuracy_score(response[test_index], y_pred)
                kappaValue = kappa(cm)
                probability = rf.predict_proba(x[test_index])
        # put all in dataframe
                rfMatrix = pd.DataFrame({
                            'run': str(i),
                            'numFeatures':len(variablesSubset),
                            'fetNames':(';'.join(map(str, variablesSubset))),
                            'precision':str(precision), 
                            'accuracy':str(accuracy),
                            'classficationAcc':classficationAcc,
                            'classesAccuracy': [(';'.join(map(str,classesAccuracy)))],
                            'classesPrecision':[(';'.join(map(str,classesPrecision)))],
                            'importances':[(';'.join(map(str,list(rf.feature_importances_))))],
                            'oobscore':str(rf.oob_score_),
                            'rfscore': str(score),
                            'classesProbability': [("; ").join([(" ").join((map(str,probability[i]))) for i in range(len(probability[0]))])],
                            'kappa' : str(kappaValue),
                            'confMatrix':[("; ").join([(" ").join((map(str,cm[i]))) for i in range(len(cm[0]))])] 	
                            })
                rfMatrix.to_csv("/home/trashtos/randomForest_run_"+str(i)+"_" + str(len(variablesSubset)) +"_.txt", index=False)
        
# ----------------------------------------------------------------------------#
# The End
# ----------------------------------------------------------------------------#