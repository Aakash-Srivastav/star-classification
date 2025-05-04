import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import RepeatedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

data = pd.read_csv('star_classification.csv')

print(data.head())
print(data.info())
print(data.describe())

data.drop(['obj_ID','run_ID','rerun_ID','cam_col', 'field_ID', 'spec_obj_ID', 'fiber_ID', 'plate', 'MJD'], axis=1, inplace=True)

sns.histplot(x=data['class'], hue=data['class'])
plt.title('Object Class Histogram')
plt.xlabel('Object Class')
plt.tight_layout()
plt.show()

data['class'] = LabelEncoder().fit_transform(data['class'])

print(data.describe())
data["class"].value_counts()

print(data[data['u'] <= 0]['u'])
print(data[data['u'] <= 0]['g'])
print(data[data['u'] <= 0]['z'])

data.drop(index=79543, inplace=True)
print(data.shape)

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot = True, fmt = ".2f", linewidths = .5, cmap='coolwarm')
plt.show()

data.hist(column=['alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
       'redshift'], bins=50, figsize=(12,12))
plt.show()

g = sns.PairGrid(data[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z','redshift']])
g.map_diag(sns.histplot)
g.map_upper(sns.scatterplot)
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['redshift'], y=data['u'], hue=data['class'], alpha=0.6)
plt.title('Redshift vs u-band Magnitude by Object Class')
plt.xlabel('Redshift')
plt.ylabel('u magnitude')
plt.legend(title='Class')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['redshift'], y=data['g'], hue=data['class'], alpha=0.6)
plt.title('Redshift vs g-band Magnitude by Object Class')
plt.xlabel('Redshift')
plt.ylabel('g magnitude')
plt.legend(title='Class')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['redshift'], y=data['r'], hue=data['class'], alpha=0.6)
plt.title('Redshift vs r-band Magnitude by Object Class')
plt.xlabel('Redshift')
plt.ylabel('r magnitude')
plt.legend(title='Class')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['redshift'], y=data['i'], hue=data['class'], alpha=0.6)
plt.title('Redshift vs i-band Magnitude by Object Class')
plt.xlabel('Redshift')
plt.ylabel('i magnitude')
plt.legend(title='Class')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['redshift'], y=data['z'], hue=data['class'], alpha=0.6)
plt.title('Redshift vs z-band Magnitude by Object Class')
plt.xlabel('Redshift')
plt.ylabel('z magnitude')
plt.legend(title='Class')
plt.show()

X = data.drop(columns=["class"], axis=1, inplace=False)
y = data[["class"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100, stratify=y
)

def Gridsearch_cv(model, params, pipeline):
        
    #GridSearch CV
    gs_clf = GridSearchCV(model, params,scoring='recall_weighted')
    gs_clf = gs_clf.fit(X_train, y_train)
    model = gs_clf.best_estimator_
    
    # Use best model and test data for final evaluation
    y_pred = model.predict(X_test)

    #Identify Best Parameters to Optimize the Model
    bestpara=str(gs_clf.best_params_)
    
    #Output Heading
    print('\nOptimized Model')
    print('\nModel Name:',str(pipeline.named_steps['clf']))

    print('\nBest Parameters:',bestpara)
    cm = confusion_matrix(y_test,y_pred)

    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()
    # print('\n', confusion_matrix(y_test,y_pred))  
    classes = ['GALAXY','STAR','QSO']
    print('\n',classification_report(y_test,y_pred,target_names=classes)) 

def logregression_model():
    #Create Pipeline

    pipeline =[]

    pipe_logreg = Pipeline([('scl', StandardScaler()),
                        ('clf', LogisticRegression(multi_class='multinomial',
                                                random_state=100,max_iter=1000))])
    pipeline.insert(0,pipe_logreg)

    # Set grid search params 

    modelpara =[]

    param_gridlogreg = {'clf__C': [0.1, 1, 10], 
                        'clf__penalty': ['l2'],
                    'clf__solver':['newton-cg', 'lbfgs']}
    modelpara.insert(0,param_gridlogreg)

    for pipeline, modelpara in zip(pipeline,modelpara):
        Gridsearch_cv(pipeline,modelpara,pipeline=pipeline)

def lda_model():

    #Create Pipeline

    pipeline =[]

    pipe_lda = Pipeline([('scl', StandardScaler()),
                        ('clf', LinearDiscriminantAnalysis())])
    pipeline.insert(0,pipe_lda)

    # Set grid search params 

    modelpara =[]

    param_gridlda = {'clf__solver':['svd','lsqr','eigen']}
    modelpara.insert(0,param_gridlda)

    for pipeline, modelpara in zip(pipeline,modelpara):
        Gridsearch_cv(pipeline,modelpara,pipeline=pipeline)
    
def qda_model():

    #Create Pipeline

    pipeline =[]
    pipe_qda = Pipeline([('scl', StandardScaler()),
                        ('clf', QuadraticDiscriminantAnalysis())])
    pipeline.insert(0,pipe_qda)

    # Set grid search params 

    modelpara =[]

    param_gridqda = {}
    modelpara.insert(0,param_gridqda)

    for pipeline, modelpara in zip(pipeline,modelpara):
        Gridsearch_cv(pipeline,modelpara,pipeline=pipeline)

def randome_forest_model():
    #Create Pipeline

    pipeline =[]

    pipe_rdf = Pipeline([('scl', StandardScaler()),
                        ('clf', RandomForestClassifier(random_state=100))])
    pipeline.insert(0,pipe_rdf)

    # Set grid search params 

    modelpara =[]

    param_gridrdf = {
                'clf__criterion':['gini','entropy'],
                'clf__n_estimators': [100]}
                # 'clf__bootstrap': [True, False]
    modelpara.insert(0,param_gridrdf)

    for pipeline, modelpara in zip(pipeline,modelpara):
        Gridsearch_cv(pipeline,modelpara,pipeline=pipeline)

def decision_tree_model():

    #Create Pipeline

    pipeline =[]

    pipe_dt = Pipeline([('scl', StandardScaler()),
                        ('clf', DecisionTreeClassifier(random_state=100))])
    pipeline.insert(0,pipe_dt)

    # Set grid search params 

    modelpara =[]

    max_depth = range(1,50)
    param_griddt = {'clf__criterion':['gini','entropy'],
                    'clf__max_depth':max_depth}
    modelpara.insert(0,param_griddt)

    for pipeline, modelpara in zip(pipeline,modelpara):
        Gridsearch_cv(pipeline,modelpara,pipeline=pipeline)

def svc_model():

    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    svc_predictions = svc_model.predict(X_test)

    print(classification_report(y_test, svc_predictions))

    cm = confusion_matrix(y_test, svc_predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

def xgb_model():

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    xgb_predictions = xgb.predict(X_test)

    print(classification_report(y_test, xgb_predictions))

    cm = confusion_matrix(y_test, xgb_predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

if __name__ == "__main__":
    logregression_model()
    lda_model()
    qda_model()
    randome_forest_model()
    decision_tree_model()
    svc_model()
    xgb_model()