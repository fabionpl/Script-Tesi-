# Import dependencies
import numpy as np
import pandas as pd
import time
import statistics
from statistics import median

# ML imports
import seaborn as sns
#from sns import barplot
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict, GridSearchCV, cross_validate, LeaveOneOut
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler

# Initialize process time list for code timing across entire program
process_time = []

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

def best_cv(model, x, y):
    accuracy=0
    X_train_best = []
    y_train_best = []
    X_test_best = []
    y_test_best = []
    #Kfold
    kf = KFold(n_splits=5) # Define the split - into 2 folds
    kf.get_n_splits() # returns the number of splitting iterations in the cross-validator

    for train_index, test_index in kf.split(x):
        # print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, np.ravel(y_train,order='C'))
        #cls_rf.fit(X_train, y_train)
        #print(X_train.shape,X_test.shape)
        #print('Accuracy: ', model.score(X_test, y_test))
        if(model.score(X_test, y_test)>= accuracy):
            accuracy = model.score(X_test, y_test)
            X_train_best = X_train
            y_train_best = y_train
            X_test_best = X_test
            y_test_best = y_test

    rf = model.fit(X_train_best, np.ravel(y_train_best,order='C'))
    print('Accuracy best for the model : ', rf.score(X_test_best, y_test_best))
    return rf, X_test_best, y_test_best


#Ricordati di modificare il file csv aggiungendo l'intestazione
df2 = pd.read_csv("misura con target_14c_06_07_2022.csv", sep=',', low_memory=False)
#print(df2.shape)

features2 = ['MCS S1-1|CatNumC','MCS S1-1|CatNumR','FC  S1-1|CR','FC  02-1|CR','FC  03-1|CR','FC  04-1|CR','FC  05-1|CR','IGC S1-1|PR','IGC 01-1|PR','IGC 02-1|PR','IGC 03-1|PR','IGC 03-2|PR','IGC 04-1|PR','IGC 04-2|PR','GS  T -1|DR','GS  T -1|PR','CH  TX-1|CR','CH  TX-1|CRlost','CH  TX-2|CR','COL TN-1|CR','COL TX-1|CR','CPS TX-1|-VR','CPS TX-1|+VR','TPS TK-1|GvmVR','MFC 02-1|CR','MFC 02-1|CRavg','MFC 02-2|CR','MFC 02-2|CRavg','MFC 04-1|CR','MFC 04-1|CRavg','MFC 04-2|CR','MFC 04-2|CRavg','MBS 01-1|VCreg0','SEQ 01-1|CntDwnC','SEQ 01-1|CntDwnR','DOSE    |AvgTrans','DOSE    |CntRate','DOSE    |CntTotGT','DOSE    |CntTotH','DOSE    |CntTotS','DOSE    |ESratio1','DOSE    |ESratio2','DOSE    |HEratio','DOSE    |LEratio','DOSE    |Transmis','INT 02-1|VR','INT 04-1|VR']
output = ['target']
X_test = []
y_test = []
X_test_pca = []
y_test_pca = []

superfeatures = list(df2.columns)
output =' target'

output2 = ' DOSE    |Transmis'
output3 = ' DOSE    |LEratio'
#output5 =' INT 04-1|VR'
#output4 = ' INT 02-1|VR'
o1 = ' DOSE    |AvgTrans'
o2 = ' DOSE    |CntRate'
o3 = ' DOSE    |CntTotGT'
o4 = ' DOSE    |CntTotH'
o5 = ' DOSE    |CntTotS'
o6 = ' DOSE    |ESratio1'
o7 = ' DOSE    |ESratio2'
o8 = ' DOSE    |HEratio'

superfeatures.remove(output)
#superfeatures.remove(output5)
#superfeatures.remove(output4)
superfeatures.remove(o1)
superfeatures.remove(o2)
superfeatures.remove(o3)
superfeatures.remove(o4)
superfeatures.remove(o5)
superfeatures.remove(o6)
superfeatures.remove(o7)
superfeatures.remove(o8)
superfeatures.remove(output2)
superfeatures.remove(output3)
#superfeatures.remove(output4)


x = df2[superfeatures]
x = x.to_numpy()
y = df2[output]
y = y.to_numpy()

cls_rf = RandomForestClassifier(random_state=1, n_jobs=-1, n_estimators=2)
start_time = time.time()
print('Model: RF, trained on: data-set initial')
rf, X_test, y_test = best_cv(cls_rf, x, y)

# Timing
elapsed_time = time.time() - start_time
print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
print('Time in seconds: ', elapsed_time)
process_time.append(elapsed_time)

# F1 score
pred = cls_rf.predict(X_test)
print('F1 accuracy: ', f1_score(pred, y_test, average='macro'))

# Classification
print(classification_report(y_test, pred))

# Plot confusion matrix of actual versus predicted labels
rf_cm = confusion_matrix(y_test, pred)
rf_cm_plt=sns.heatmap(rf_cm.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")
plt.xlabel('Actual label')
plt.ylabel('Predicted label')
plt.title("Valid")
plt.show()

# Feature Importance
feature_imp = pd.Series(rf.feature_importances_, index=superfeatures).sort_values(ascending = False)
sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Punteggio di importanza delle caratteristiche')
plt.ylabel('Caratteristiche')
plt.title('Visualizzazione caratteristiche più importanti')
plt.tight_layout()
plt.savefig('Features_rf_tot.png')
plt.show()

# Fit model using PCA
#pca = PCA(.26) #1 number of components
start_time = time.time()
pca = PCA(.90)
x_pca = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x_pca)

# Timing
elapsed_time = time.time() - start_time
print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
print('Time in seconds: ', elapsed_time)
process_time.append(elapsed_time)
#print(pca.n_components_)
#print(pca.explained_variance_ratio_)

#Explained variance plot
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance')
#plt.show()

# Fit the model
start_time = time.time()
print('Model: RF, trained on: data-set preprocessed with PCA')
rf2, X_test_pca, y_test_pca = best_cv(cls_rf, principalComponents, y)

# Timing
elapsed_time = time.time() - start_time
print('Formatted time: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
print('Time in seconds: ', elapsed_time)
process_time.append(elapsed_time)

# Predict a second time with principal components
pred2 = rf2.predict(X_test_pca)
print('F1 accuracy: ', f1_score(pred2, y_test_pca, average='macro'))

# Contribute of variables on Dim 1 (2,3...)
eigenvalues=pca.components_
#print(pca.n_components_)

PC1=abs(eigenvalues[0,:])
comp = pd.Series(PC1, index=superfeatures).sort_values(ascending = False)
sns.barplot(x = comp, y = comp.index)
plt.xlabel('Features')
plt.ylabel('Contribute')
plt.title('Contribute Features Dim1 PCA')
plt.tight_layout()
plt.savefig('Contribute Features Dim1 PCA')
plt.show()

# Explained variance plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# Scatter Plot
pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
plt.legend('')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.show()
