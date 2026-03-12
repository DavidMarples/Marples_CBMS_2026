# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx
import xgboost as xgb
import lightgbm as lgb
import shap

from pickle import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.feature_selection import RFE,RFECV
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_squared_error

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.metrics import r2_score
from scipy.stats import linregress

#%% Functions that will be needed later

def prior_correct(p, pi_train=0.25, pi_test=1/21):
    '''
    Takes a set of predicted probabilities (by default for the 3:1 data)
    and adjusts the probabilities for the test control:positive ratios (by default 20:1)

    Parameters
    ----------
    p : array of probabilities.
    pi_train : The fraction that were positive in the training data. The default is 0.25.
    pi_test : The fraction that are expected to be positive in the test data. The default is 1/21.

    Returns
    -------
    res : the array of adjusted probabilities.

    '''
    p = np.clip(p, 1e-12, 1 - 1e-12) # numerical safety
    num = (pi_test / pi_train) * p
    den = num + ((1 - pi_test) / (1 - pi_train)) * (1 - p)
    res = num/den
    return res


def CalibCurve(preds,labels,title):
    '''
    Parameters
    ----------
    preds : Output from a SKLearn predict_proba() function, assumed to have 2 values per row (i.e. prob(0) and prob(1))
    labels: A column of a pandas dataframe containing the true labels
    title: string describing the dataset, to go in the graph title

    Returns
    -------
    list of the percentage of each decile which were actually positive, plus the number of rows in each probability decile
    '''
    preds = preds.tolist()
    preds = [x[1] for x in preds]
    data = pd.DataFrame(preds,columns=["Probs"])
    data["Results"]=labels.tolist()
    tenths = [0.05,0.15,0.25,0.33,0.45,0.55,0.65,0.75,0.85,0.95]
    calib = []
    numbers = []
    non_zeroes = 0
    for x in range(10):
        minx = x/10
        maxx = minx + 0.1
        subset = data[(data["Probs"]>minx) & (data["Probs"]<=maxx)]
        n = len(subset)
        #print(n)
        numbers.append(n)
        if n>0:
            calib.append(subset["Results"].sum()/n)
            non_zeroes += 1
        else:
            calib.append(0)
    plt.scatter(tenths, calib)
    plt.title(f"Calibration Curve by Deciles for {title}")
    plt.xlabel("Predicted probabilities")
    plt.ylabel("Actual fraction of those that are positive")
    try:
        m,c = np.polyfit(tenths[:non_zeroes],calib[:non_zeroes],1)
        plt.axline(xy1=(0, c), slope=m, label=f'$y = {m:.2f}x {c:+.2f}$')
        plt.legend()
        plt.savefig(f"d:/Project/figures/{title}.png",dpi=300,format="png")
    except:
        print("Linear regression failed!")
        print(calib)
            
    #line = m*tenths + c
    #plt.plot(tenths,line)
    plt.show()
    return calib,numbers


def CalibCurve2(preds,labels,title):
    '''
    Parameters
    ----------
    preds : Output from a SKLearn predict_proba() function, assumed to have 2 values per row (i.e. prob(0) and prob(1))
    labels: A column of a pandas dataframe containing the true labels
    title: string describing the dataset, to go in the graph title

    Returns
    -------
    list of the percentage of each decile which were actually positive, plus the number of rows in each probability decile
    '''
    preds = preds.tolist()
    #preds = [x[1] for x in preds]
    data = pd.DataFrame(preds,columns=["Probs"])
    data["Results"]=labels.tolist()
    tenths = [0.05,0.15,0.25,0.33,0.45,0.55,0.65,0.75,0.85,0.95]
    calib = []
    numbers = []
    non_zeroes = 0
    for x in range(10):
        minx = x/10
        maxx = minx + 0.1
        subset = data[(data["Probs"]>minx) & (data["Probs"]<=maxx)]
        n = len(subset)
        #print(n)
        numbers.append(n)
        if n>0:
            calib.append(subset["Results"].sum()/n)
            non_zeroes += 1
        else:
            calib.append(0)
    plt.scatter(tenths, calib)
    plt.title(f"Calibration Curve by Deciles for {title}")
    plt.xlabel("Predicted probabilities")
    plt.ylabel("Actual fraction of those that are positive")
    try:
        m,c = np.polyfit(tenths[:non_zeroes],calib[:non_zeroes],1)
        plt.axline(xy1=(0, c), slope=m, label=f'$y = {m:.2f}x {c:+.2f}$')
        plt.legend()
    except:
        print("Linear regression failed!")
        print(calib)
            
    #line = m*tenths + c
    #plt.plot(tenths,line)
    plt.show()
    return calib,numbers

def CalibCurveUntitled(preds,labels,title):
    '''
    As above, but doesn't put a title on the plot, so we can use it in a paper'
    
    '''
    preds = preds.tolist()
    preds = [x[1] for x in preds]
    data = pd.DataFrame(preds,columns=["Probs"])
    data["Results"]=labels.tolist()
    tenths = [0.05,0.15,0.25,0.33,0.45,0.55,0.65,0.75,0.85,0.95]
    calib = []
    numbers = []
    non_zeroes = 0
    for x in range(10):
        minx = x/10
        maxx = minx + 0.1
        subset = data[(data["Probs"]>minx) & (data["Probs"]<=maxx)]
        n = len(subset)
        #print(n)
        numbers.append(n)
        if n>0:
            calib.append(subset["Results"].sum()/n)
            non_zeroes += 1
        else:
            calib.append(0)
    plt.scatter(tenths, calib)
    #plt.title(f"Calibration Curve by Deciles for {title}")
    plt.xlabel("Predicted probabilities")
    plt.ylabel("Actual fraction of those that are positive")
    try:
        m,c = np.polyfit(tenths[:non_zeroes],calib[:non_zeroes],1)
        plt.axline(xy1=(0, c), slope=m, label=f'$y = {m:.2f}x {c:+.2f}$')
        plt.legend()
        plt.savefig(f"d:/Project/figures/{title}.png",dpi=300,format="png")
    except:
        print("Linear regression failed!")
        print(calib)
            
    #line = m*tenths + c
    #plt.plot(tenths,line)
    plt.show()
    return calib,numbers

def save_shap_beeswarm(shap_values, filename, **kwargs):
    """
    Generate a SHAP beeswarm plot and save it to file,
    without displaying it or interfering with Spyder's backend.
    Generated by Copilot, adapted by me
    (But actually it's not really needed, since now shap.plot includes options to ignore plt.show)

    Parameters
    ----------
    shap_values : shap.Explanation or array-like
        The SHAP values to plot.
    filename : str
        Output file path (e.g., "beeplot.png").
    **kwargs :
        Any additional arguments passed to plt.savefig(), e.g.
        dpi=300, bbox_inches='tight', format='png'

    Returns
    -------
    None
    """

    # Save original plt.show
    original_show = plt.show
    # Replace show with a no-op so SHAP doesn't clear the figure
    plt.show = lambda *args, **kw: None
    # Create the SHAP plot
    shap.plots.beeswarm(shap_values,max_display=9,group_remaining_features=False)
    # Save the current figure
    plt.savefig(filename, **kwargs)
    # Close the figure to avoid clutter
    plt.close()
    # Restore the original plt.show
    plt.show = original_show
    

def Do_CV_Results(model,Xdata,Y):
    # Expects 
    # a SKLearn model with the usual options available, 
    # a PANDAS dataframe containing the data
    # a PANDAS dataframe containing the labels
    skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    CVresults_list=[]
    for i,(tr_id,te_id) in enumerate(skf.split(Xdata,Y)):
        print(f"Doing split {i}")
        Xtr,Ytr = Xdata.iloc[tr_id],Y.iloc[tr_id]
        Xte,Yte = Xdata.iloc[te_id],Y.iloc[te_id]
        
        # I'm going to remerge the training data and labels, so we can resample to 3:1, then split them up again
        Xtr["label"]=Ytr
        
        emerg = Xtr[Xtr["label"] > 0]
        no_emerg = Xtr[Xtr["label"] == 0].sample(n=len(emerg)*3,random_state = 10)
        X3tr = pd.concat([emerg,no_emerg])
        Y3tr = X3tr["label"].astype("int32")
        Y3tr = np.sign(Y3tr)
        X3tr.drop(["label"],axis=1,inplace=True)
        
        # Now we'll train and run a standard scaler over the data:
        scaler = StandardScaler()
        scaler.fit(X3tr)
        Xtr = scaler.transform(X3tr)
        Xte = scaler.transform(Xte)
    
        model.fit(X3tr,Y3tr)
        train_preds = model.predict(X3tr)
        test_preds = model.predict(Xte)
        
        prec,recall,f1,support = prfs(Yte,test_preds,average="macro")
        acc = accuracy_score(Yte,test_preds)
        bacc = balanced_accuracy_score(Yte,test_preds)
        print(f"precision {prec}, recall {recall},f1 {f1}, accuracy {acc},balanced accuracy {bacc}")
        print(prec,recall,f1,acc,bacc)
        CVresults_list.append([prec,recall,f1,support,acc,bacc])
        
def BinSearchThreshold(preds,n,min=0.0,max=1.0):
    '''
    Function takes a set of predicted probabilities, and finds the threshold
    such that about n positives (within 1%). Works recursively.
    Parameters
    ----------
    preds : array of predicted probabilities
    n : integer defining how many positives are required
    min : minimum probability to consider The default is 0.0.
    max : maximum probability to consider The default is 1.0.

    Returns
    -------
    The threshold value, as a float, that will yield n positives.

    '''
    thresh = (min + max)/2.0
    npred = np.sum(preds>thresh)
    if abs(npred-n)< n/100:
        #print(thresh,abs(npred-n))
        return thresh
    if npred>n:
        # we're getting too many, so we need a higher threshold
        return BinSearchThreshold(preds,n,min=thresh, max=max)
    else:
        # we're getting too few: we need a lower threshold
        return BinSearchThreshold(preds,n,min=min, max=thresh)


#%%
merged=pd.read_csv("X:/oncology_SJUHBW/Admissions Project/Data Extract/post_covid_sex_reg_diag.csv")


#%% Alternative, using new version with regimen and diagnosis already done...

modelcolsTX = ['BinnedAgeAtCycle', 'MaxDays', 'GFR', 'SurfaceArea', 'Weight', 'Height',
             'ALB2_val', 'ALT1_done', 'ALT1_val', 'LDH2_done', 'LDH2_val','BIL1_done', 'BIL1_val', 'ALP2_done',
             'ALP2_val', 'EGFR2_done','EGFR2_val', 'FT41_done', 'FT41_val', 'TSH1_done', 'TSH1_val',
             'NA2_done', 'NA2_val', 'K2_done', 'K2_val', 'URE2_done', 'URE2_val',
             'CRE2_done', 'CRE2_val', 'PLT_done', 'PLT_val', 'HB2_done', 'HB2_val',
             'WBC_done', 'WBC_val', 'PCV_done', 'PCV_val', 'INR_done', 'INR_val',
             'APTT1_done', 'APTT1_val', 'PT1_done', 'PT1_val', 'FIBD1_done',
             'FIBD1_val', 'CRP1_done', 'CRP1_val', 'RBC_done', 'RBC_val', 'RDW_done',
             'RDW_val', 'MCH_done', 'MCH_val', 'MCV_done', 'MCV_val', 'HYPO_done',
             'HYPO_val', 'LYM_done', 'LYM_val', 'NEU_done', 'NEU_val', 'MON_done',
             'MON_val', 'EOS_done', 'EOS_val', 'BAS_done', 'BAS_val', 'LUC_done',
             'LUC_val', 'C1251_done', 'C1251_val', 'C1531_done', 'C1531_val',
             'C1991_done', 'C1991_val', 'CEA1_done', 'CEA1_val',
             'VINCRISTINE_given','VINCRISTINE_dose','METHOTREXATE_given','METHOTREXATE_dose','MERCAPTOPURINE_given','MERCAPTOPURINE_dose','ETOPOSIDE_given','ETOPOSIDE_dose',
             'DOXORUBICIN_given','DOXORUBICIN_dose','IFOSFAMIDE_given','IFOSFAMIDE_dose','CISPLATIN_given','CISPLATIN_dose','CARBOPLATIN_given','CARBOPLATIN_dose',
             'CYCLOPHOSPHAMIDE_given','CYCLOPHOSPHAMIDE_dose','CYTARABINE_given','CYTARABINE_dose','ASPARAGINASE_given','ASPARAGINASE_dose',
             'DACTINOMYCIN_given','DACTINOMYCIN_dose','EPIRUBICIN_given','EPIRUBICIN_dose','PACLITAXEL_given','PACLITAXEL_dose','GEMCITABINE_given','GEMCITABINE_dose',
             'VINBLASTINE_given','VINBLASTINE_dose','CHLORAMBUCIL_given','CHLORAMBUCIL_dose','PROCARBAZINE_given','PROCARBAZINE_dose','LOMUSTINE_given','LOMUSTINE_dose',
             'DACARBAZINE_given','DACARBAZINE_dose','BLEOMYCIN_given','BLEOMYCIN_dose','PEG_given','PEG_dose','DAUNORUBICIN_given','DAUNORUBICIN_dose',
             'IRINOTECAN_given','IRINOTECAN_dose','INTERFERON_given','INTERFERON_dose','FLUDARABINE_given','FLUDARABINE_dose','RITUXIMAB_given','RITUXIMAB_dose',
             'IDARUBICIN_given','IDARUBICIN_dose','IMATINIB_given','IMATINIB_dose','TOPOTECAN_given','TOPOTECAN_dose','VINORELBINE_given','VINORELBINE_dose',
             '5-FLUOROURACIL_given','5-FLUOROURACIL_dose', 'MIFAMURTIDE_given','MIFAMURTIDE_dose','TRASTUZUMAB_given','TRASTUZUMAB_dose','OXALIPLATIN_given','OXALIPLATIN_dose',
             'CAPECITABINE_given','CAPECITABINE_dose','SUNITINIB_given','SUNITINIB_dose','OCTREOTIDE_given','OCTREOTIDE_dose','AZACITIDINE_given','AZACITIDINE_dose',
             'BORTEZOMIB_given','BORTEZOMIB_dose','MITOMYCIN_given','MITOMYCIN_dose','TEMOZOLOMIDE_given','TEMOZOLOMIDE_dose','TAMOXIFEN_given','TAMOXIFEN_dose',
             'DARBEPOETIN_given','DARBEPOETIN_dose','HYDROXYCARBAMIDE_given','HYDROXYCARBAMIDE_dose','DOCETAXEL_given','DOCETAXEL_dose','LANREOTIDE_given','LANREOTIDE_dose',
             'THALIDOMIDE_given','THALIDOMIDE_dose','ZOLEDRONIC_given','ZOLEDRONIC_dose','SORAFENIB_given','SORAFENIB_dose','CETUXIMAB_given','CETUXIMAB_dose',
             'FULVESTRANT_given','FULVESTRANT_dose','ERIBULIN_given','ERIBULIN_dose','BEVACIZUMAB_given','BEVACIZUMAB_dose','BENDAMUSTINE_given','BENDAMUSTINE_dose',
             'LENALIDOMIDE_given','LENALIDOMIDE_dose','DENOSUMAB_given','DENOSUMAB_dose','PEMETREXED_given','PEMETREXED_dose','ERLOTINIB_given','ERLOTINIB_dose',
             'GOSERELIN_given','GOSERELIN_dose','DASATINIB_given','DASATINIB_dose','EVEROLIMUS_given','EVEROLIMUS_dose','IPILIMUMAB_given','IPILIMUMAB_dose',
             'NILOTINIB_given','NILOTINIB_dose','PAZOPANIB_given','PAZOPANIB_dose','LEUPRORELIN_given','LEUPRORELIN_dose','VEMURAFENIB_given','VEMURAFENIB_dose',
             'ABIRATERONE_given','ABIRATERONE_dose','BCG_given','BCG_dose','EPOETIN_given','EPOETIN_dose','DABRAFENIB_given','DABRAFENIB_dose','IBRUTINIB_given','IBRUTINIB_dose',
             'RUXOLITINIB_given','RUXOLITINIB_dose','ENZALUTAMIDE_given','ENZALUTAMIDE_dose','PERTUZUMAB_given','PERTUZUMAB_dose','TRAMETINIB_given','TRAMETINIB_dose',
             'CARFILZOMIB_given','CARFILZOMIB_dose','OLAPARIB_given','OLAPARIB_dose','FILGRASTIM_given','FILGRASTIM_dose','DARATUMUMAB_given','DARATUMUMAB_dose',
             'HYDROCORTISONE_given','HYDROCORTISONE_dose','MESNA_given','MESNA_dose','METHYLPREDNISOLONE_given','METHYLPREDNISOLONE_dose','PREDNISOLONE_given','PREDNISOLONE_dose',
             'PEGFILGRASTIM_given','PEGFILGRASTIM_dose','NIVOLUMAB_given','NIVOLUMAB_dose','PEMBROLIZUMAB_given','PEMBROLIZUMAB_dose','PAMIDRONATE_given','PAMIDRONATE_dose',
             'PALBOCICLIB_given','PALBOCICLIB_dose','ATEZOLIZUMAB_given','ATEZOLIZUMAB_dose','DEXAMETHASONE_given','DEXAMETHASONE_dose','ONDANSETRON_given','ONDANSETRON_dose',
             'NYSTATIN_given','NYSTATIN_dose','ACICLOVIR_given','ACICLOVIR_dose','LANSOPRAZOLE_given','LANSOPRAZOLE_dose','CO-TRIMOXAZOLE_given','CO-TRIMOXAZOLE_dose', 
             'PriorEmergencies','SubsequentEmergencies',
             # 'EmergencyDate','TimeToEmerg','Duration','N_Emergencies','N_Adms','WksToEm'] 
             'T1Diabetes', 'T2Diabetes', 'OtherDiabetes', 'MI', 'CCF', 'CAD', 'HPTN',
             'Arrhy', 'Venous', 'Varicose', 'Thromb', 'Cardiomyopathy', 'Resp',
             'Asthma', 'Restrictive', 'COPD', 'Malabsorption', 'IBD', 'PUD',
             'Pancreatitis', 'Liver', 'Renal', 'Paraplegia', 'Neuromuscular',
             'Parkinsons', 'Demyelination', 'MND', 'TIA', 'Stroke', 'Dementia',
             'HIV', 'Rheum', 'RA', 'Psoriatic', 'Gout', 'AS', 'Malignancy',
             'Obesity', 'Hyperlipidaemia', 'PAD', 'Spinal', 'N_comorbs','SexNum','NoKHome',
             'RegimenScore','DiagScore']

X = merged[modelcolsTX]

Y = X["SubsequentEmergencies"].astype("int32")
Y = np.sign(Y)   # We'll convert a number of subsequent admissions to a simple 0/1 flag of whethere there were any

#%% Separated out so I can re-run just this bit to regenerate them...
Xtrain_unscaled, Xtest_unscaled, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,random_state = 42)



#%% Make a subset with only a BIT more controls
emerg = Xtrain_unscaled[Xtrain_unscaled["SubsequentEmergencies"] > 0]
no_emerg = Xtrain_unscaled[Xtrain_unscaled["SubsequentEmergencies"]== 0].sample(n=len(emerg)*3,random_state = 10)

print(len(emerg),len(no_emerg))
X3 = pd.concat([emerg,no_emerg])
Y3 = X3["SubsequentEmergencies"].astype("int32")
Y3 = np.sign(Y3)


#%% ...and drop the  LABEL (subsequentemergencies)!
Xtrain_unscaled.drop(["SubsequentEmergencies"],axis=1,inplace=True)
Xtest_unscaled.drop(["SubsequentEmergencies"],axis=1,inplace=True)

Xtrain_unscaled = Xtrain_unscaled.astype("float64")
Xtest_unscaled = Xtest_unscaled.astype("float64")

X3.drop(["SubsequentEmergencies"],axis=1,inplace=True)
X3 = X3.astype("float64")

#%% Scale all columns to mean 0 SD 1 Note we're putting this into another 
# set of dataframes, leaving the unscaled ones untouched

Xtrain_unscaled.fillna(0,inplace=True)
Xtest_unscaled.fillna(0,inplace=True)

#Here we scale everything
scaler = StandardScaler()
scaler.fit(Xtrain_unscaled)
Xtrain = scaler.transform(Xtrain_unscaled)
Xtest = scaler.transform(Xtest_unscaled)

#%% Do the same for the smaller X3 set:
X3.fillna(0,inplace=True)

#Here we scale everything
scaler3 = StandardScaler()
scaler3.fit(X3)
X3train = scaler3.transform(X3)

#%% Now I'm going to generate some models, using the X3train data. optimised using grid search

Ytest = np.sign(Ytest)
Xtest = scaler.transform(Xtest_unscaled)
print(len(Xtest),len(Ytest))
# We'll try to optimise a random forest... 
RF_clf = RandomForestClassifier(random_state=0) # max_depth=5, n_estimators=300)
param_grid = {"max_depth": [3,5,10,20],
              "n_estimators": [100,300,1000,2000]}
RF_opt = GridSearchCV(RF_clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
RF_opt.fit(X3train, Y3)
print("Results from random forest:")
print(RF_opt.best_params_)
RFout = RF_opt.predict(Xtest)
print(classification_report(Ytest, RFout))

#%% ... and a Logistic Regression model
LR_clf = LogisticRegression(random_state=0) #penalty='l2',max_iter=1000,, C = 100.0,solver='liblinear') #,l1_ratio = 1.0

param_grid = {"max_iter": [500,1000,2000],
              "penalty": ["l1","l2"],
              "C": [1.0,10.0,100.0],
              "solver": ['liblinear',"saga"]}
LR_opt = GridSearchCV(LR_clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
LR_opt.fit(X3train,Y3)
print("Results from logistic regression:")
print(LR_opt.best_params_)
#%%
LRout = LR_opt.predict_proba(Xtest)
print(classification_report(Ytest, (LRout >= 0.5)))




#%% Let's try just using LASSO to reduce the number of parameters (using the scaled Xtrain and Xtest sets from above)
workingfields = X.columns
clf_full = LogisticRegression(penalty='l1',max_iter=5000,random_state=0,  C = 1.0, solver='liblinear',class_weight='balanced') #,l1_ratio = 1.0
#clf_full = LogisticRegression(penalty='elasticnet',max_iter=2000,random_state=0,  C = 1.0, solver='saga',l1_ratio = 0.5)
#clf_full.fit(Xtrain[:,:25], Ytrain)
clf_full.fit(Xtrain, Ytrain)
#LR_full_out = clf_full.predict(Xtrain[:,:25])
LR_full_out = clf_full.predict(Xtrain)

print(f"The training accuracy is {1-sum(abs(LR_full_out-Ytrain))/len(Ytrain)}")

#LR_full_out2 = clf_full.predict(Xtest[:,:25])
LR_full_out2 = clf_full.predict(Xtest)

print(f"The test accuracy is {1-sum(abs(LR_full_out2-Ytest))/len(Ytest)}")

coeffs = clf_full.coef_[0]
importances = []
for n in range(len(coeffs)):
    if coeffs[n] > 0: importances.append((abs(coeffs[n]),workingfields[n]))
importances.sort(reverse = True)                       
print(f"There are {len(importances)} non-zero coefficients")
print(importances[:20])
print(f"There were {sum(LR_full_out2)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print(classification_report(Ytrain, LR_full_out))
print(classification_report(Ytest, LR_full_out2)) 
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, LR_full_out2))
       

#%% Now we'll use the same model to generate probabilities rather than classification

LR_train_probs = clf_full.predict_proba(Xtrain)
LR_test_probs = clf_full.predict_proba(Xtest)
LR_test=LR_test_probs.tolist()
LR_test1 = [x[1] for x in LR_test]
plt.hist(LR_test1)
plt.title("Distribution of probabilities for test data")
plt.xlabel("Probability of emergency admission")
plt.ylabel("Number of cases")
plt.show()

probdf = pd.DataFrame(LR_test1,columns=["Probs"])
probdf["Results"]=Ytest.tolist()
probdf.info() 

fpr, tpr, thresh = roc_curve(Ytest, LR_test1)
plt.plot(fpr, tpr)
plt.title('AUC for test LR data = ' + str(roc_auc_score(Ytest, LR_test1)))
plt.show()

#%% Now we can pull out blocks where the probabilities lie within a given range, and see what fraction are 1's

#%%
results,numbers = CalibCurve(LR_test_probs,Ytest,"Logistic Regression Model")

#%% Let's try with the smaller subset
workingfields = Xtrain_unscaled.columns
clf3_full = LogisticRegression(penalty='l1',max_iter=5000,random_state=0,  C = 1.0, solver='liblinear',class_weight='balanced') #,l1_ratio = 1.0
clf3_full.fit(X3train, Y3)
LR3_full_out = clf3_full.predict(X3train)

print(f"The training accuracy is {1-sum(abs(LR3_full_out-Y3))/len(Y3)}")

LR_full_out2 = clf3_full.predict(Xtest)

print(f"The test accuracy is {1-sum(abs(LR_full_out2-Ytest))/len(Ytest)}")

coeffs3 = clf3_full.coef_[0]
importances = []
for n in range(len(coeffs3)):
    if coeffs3[n] > 0: importances.append((abs(coeffs3[n]),workingfields[n]))
importances.sort(reverse = True)                       
print(f"There are {len(importances)} non-zero coefficients")
print(importances[:20])
print(f"There were {sum(LR_full_out2)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print(classification_report(Y3, LR3_full_out))
print(classification_report(Ytest, LR_full_out2)) 
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, LR_full_out2))
       

#%% Now we'll use the same model to generate probabilities rather than classification

LR3_train_probs = clf3_full.predict_proba(X3train)
LR3_test_probs = clf3_full.predict_proba(Xtest)
LR3_test=LR3_test_probs.tolist()
LR3_test1 = [x[1] for x in LR3_test]
plt.hist(LR3_test1)
plt.title("Distribution of probabilities for test data")
plt.xlabel("Probability of emergency admission")
plt.ylabel("Number of cases")
plt.show()

probdf = pd.DataFrame(LR3_test1,columns=["Probs"])
probdf["Results"]=Ytest.tolist()
probdf.info() 

fpr, tpr, thresh = roc_curve(Ytest, LR3_test1)
plt.plot(fpr, tpr)
plt.title('AUC for test LR data = ' + str(roc_auc_score(Ytest, LR3_test1)))
plt.show()

results,numbers = CalibCurve(LR3_test_probs,Ytest,"Logistic Regression Model without correction")
  
results,numbers = CalibCurve(prior_correct(LR3_test_probs),Ytest,"Logistic Regression Model with correction")
    

#%% Try SVC with the 3:1 data
svc3_clf = SVC(C = 1.0,kernel='rbf', class_weight='balanced') 
svc3_clf.fit(X3train,Y3)

train3_res = svc3_clf.predict(X3train)
test3_res = svc3_clf.predict(Xtest)

print(classification_report(Y3, train3_res))
print(classification_report(Ytest, test3_res))
print(f"There were {sum(test3_res)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print("Confusion matrix for test data and 3:1 training")
print(confusion_matrix(Ytest, test3_res))

# As before, we'll need a new model...
svc3_prob = SVC(C = 1.0,kernel='rbf', class_weight='balanced',probability=True) 
svc3_prob.fit(X3train,Y3)

SC3_train_probs = svc3_prob.predict_proba(X3train)
SC3_test_probs = svc3_prob.predict_proba(Xtest)
SC3_test=SC3_test_probs.tolist()
SC3_test1 = [x[1] for x in SC3_test]
plt.hist(SC3_test1)
plt.title("Distribution of probabilities for test data after 3:1 training for SVC")
plt.xlabel("Probability of emergency admission")
plt.ylabel("Number of cases")
plt.show()

fpr, tpr, thresh = roc_curve(Ytest, SC3_test1)
plt.plot(fpr, tpr)
plt.title('AUC for test SVM data after 3:1 training = ' + str(roc_auc_score(Ytest, SC3_test1)))
plt.show()

results,numbers = CalibCurve(SC3_test_probs,Ytest,"SVC Model and 3:1 Training")


#%% Try Random Forest with 3:1 training data, and 20:1 test data

RF3_clf = RandomForestClassifier(random_state=0, max_depth=10, n_estimators=1000, class_weight='balanced')
RF3_clf.fit(X3train,Y3)
RF3train_res = RF3_clf.predict(X3train)
RF3test_res = RF3_clf.predict(Xtest)
print("Results from random forest: 3:1 training first")
print(classification_report(Y3, RF3train_res))
print("Test results")
print(classification_report(Ytest, RF3test_res))
print(f"There were {sum(RF3test_res)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, RF3test_res))


RF3_train_probs = RF3_clf.predict_proba(X3train)
RF3_test_probs = RF3_clf.predict_proba(Xtest)
RF3_test=RF3_test_probs.tolist()
RF3_test1 = [x[1] for x in RF3_test]
plt.hist(RF3_test1)
plt.title("Distribution of probabilities for 20:1 test data after 3:1 training")
plt.xlabel("Probability of emergency admission")
plt.ylabel("Number of cases")
plt.show()



fpr, tpr, thresh = roc_curve(Ytest, RF3_test1)
plt.plot(fpr, tpr)
plt.title('AUC for test (20:1) RF data = ' + str(roc_auc_score(Ytest, RF3_test1)))
plt.show()

results,numbers = CalibCurve(RF3_test_probs,Ytest,"RF Model")
#%% Let's see what correction does to the calibration curve

results,numbers = CalibCurve(prior_correct(RF3_test_probs),Ytest,"RF Model")


#%% Let's try training the NN with the 3:1 data, and then test on 20:1

nn_clf2 = MLPClassifier(activation= 'relu', hidden_layer_sizes= (315,100,50), max_iter= 2000, solver= 'adam')
nn_clf2.fit(X3train,Y3)
nntrain3_res = nn_clf2.predict(X3train)
nntest3_res = nn_clf2.predict(Xtest)
print("Results from Neural net: training first")
print(classification_report(Y3, nntrain3_res))
print("Test results")
print(classification_report(Ytest, nntest3_res))
print(f"There were {sum(nntest3_res)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, nntest3_res))

nn_train3_probs = nn_clf2.predict_proba(X3train)
nn_test3_probs = nn_clf2.predict_proba(Xtest)
results,numbers = CalibCurve(nn_test3_probs,Ytest,"Neural Net Model uncorrected")
results,numbers = CalibCurve(prior_correct(nn_test3_probs),Ytest,"Neural Net Model corrected")

nn_test3=nn_test3_probs.tolist()
nn_test31 = [x[1] for x in nn_test3]
plt.hist(nn_test31)
plt.title("Distribution of probabilities for test data")
plt.xlabel("Probability of emergency admission")
plt.ylabel("Number of cases")
plt.show()

probdf = pd.DataFrame(nn_test31,columns=["Probs"])
probdf["Results"]=Ytest.tolist()
probdf.info() 
#%%
fpr, tpr, thresh = roc_curve(Ytest, nn_test31)
plt.plot(fpr, tpr)
plt.title('AUC for test NN data = ' + str(roc_auc_score(Ytest,nn_test31)))
plt.show()

#%% OK, let try running with an XGB classifier Let's try with the 3:1 data
xgb_model = xgb.XGBClassifier(booster = 'gbtree', objective="binary:logistic", random_state=42, scale_pos_weight=3)
#xgb_model = xgb.XGBClassifier(booster = 'gblinear', objective="binary:logistic", random_state=42, class_weight='balanced')

xgb_model.fit(X3train, Y3)

ytrain_pred = xgb_model.predict(X3train)
ytest_pred = xgb_model.predict(Xtest)

print("For the training set:")                             
print(classification_report(Y3, ytrain_pred))
print("Confusion matrix for training data")
print(confusion_matrix(Y3, ytrain_pred))
print("For the test set:")                             
print(classification_report(Ytest, ytest_pred))
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, ytest_pred))
print(f"There were {sum(ytest_pred)} predicted and {sum(Ytest)} actual admissions in the test set.")                             

#%% Let's try making a regressor, and seeing what the calibration looks like


xgb_reg_model = xgb.XGBRegressor(booster = 'gbtree',objective="reg:linear", random_state=42, scale_pos_weight=3)

xgb_reg_model.fit(Xtrain, Ytrain)

xgb_train_probs = xgb_model.predict_proba(Xtrain)
xgb_test_probs = xgb_model.predict_proba(Xtest)
results,numbers = CalibCurve(xgb_test_probs,Ytest,"XGBoost Model uncorrected")
results,numbers = CalibCurve(prior_correct(xgb_test_probs),Ytest,"XGBoost Model corrected")

xgb_test=xgb_test_probs.tolist()
xgb_test1 = [x[1] for x in xgb_test]
plt.hist(xgb_test1)
plt.title("Distribution of probabilities for XGB test data")
plt.xlabel("Probability of emergency admission")
plt.ylabel("Number of cases")
plt.show()

mse=mean_squared_error(Ytest, xgb_test1)


fpr, tpr, thresh = roc_curve(Ytest, xgb_test1)
plt.plot(fpr, tpr)
plt.title('AUC for test XGB data = ' + str(roc_auc_score(Ytest, xgb_test1)))
plt.show()
#%%
xgb.plot_importance(xgb_model,max_num_features=20,importance_type='weight')
#%%
feature_names = Xtrain_unscaled.columns
importance_types = ['weight','gain','cover']
for it in importance_types:
    importance_dic = xgb_model.get_booster().get_score(importance_type=it)
    keys = list(importance_dic.keys())
    vals = list(importance_dic.values())
    keystext = [feature_names[int(n[1:])] for n in keys]
    importances = pd.DataFrame(data=vals, index=keystext, columns=["score"]).sort_values(by = "score", ascending=False)
    importances.nlargest(40, columns="score").plot(kind='barh', figsize = (10,10),title=f'Highest importances based on {it}') ## plot top 40 features
    #importances.nsmallest(40, columns="score").plot(kind='barh', figsize = (10,10),title=f'Lowest importances based on {it}') ## plot bottom 40 features
      



#%% LightGBM with the 3:1 training data

lgb_clf3 = lgb.LGBMClassifier()
lgb_clf3.fit(X3train,Y3)
LGB3train_res = lgb_clf3.predict(X3train)
LGB3test_res = lgb_clf3.predict(Xtest)
print("Results from LightGBM with 3:1 training data: training first")
print(classification_report(Y3, LGB3train_res))
print("Test results with 3:1 training")
print(classification_report(Ytest, LGB3test_res))
print(f"There were {sum(LGB3test_res)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print("Confusion matrix for test data with 3:1 training")
print(confusion_matrix(Ytest, LGB3test_res))

LGB3train_probs = lgb_clf3.predict_proba(X3train)
LGB3test_probs = lgb_clf3.predict_proba(Xtest)
results,numbers = CalibCurve(LGB3test_probs,Ytest,"LGB_3:1_training")
LGB3_test=LGB3test_probs.tolist()
LGB3_test1 = [x[1] for x in LGB3_test]

fpr, tpr, thresh = roc_curve(Ytest, LGB3_test1)
plt.plot(fpr, tpr)
plt.title('AUC for test LightGBM data with 3:1 training= ' + str(roc_auc_score(Ytest,LGB3_test1)))
plt.show()

#%% Calbration curve corrected
results,numbers = CalibCurve(prior_correct(LGB3test_probs),Ytest,"LGB_3")


#%% OK, let's have a play with SHAP

# import shap  #Up the top now


flist = (Xtrain_unscaled.columns).tolist()
flist[flist.index("ALB2_val")]="Plasma Albumin"
flist[flist.index("NA2_val")]="Plasma Sodium"
flist[flist.index("WBC_val")]="White cell count"
flist[flist.index("HB2_val")]="Haemoglobin"
flist[flist.index("ALP2_val")]="Alkaline Phosphatase"
flist[flist.index("RDW_val")]="Red Cell Width Distribution"
flist[flist.index("N_comorbs")]="Number of Comorbidities"
flist[flist.index("PriorEmergencies")]="Prior Emergencies"
flist[flist.index("RegimenScore")]="Regimen Risk Score"



#%%
#explainer = shap.Explainer(lgb_clf3, Xtrain, feature_names=Xtrain_unscaled.columns)
explainer = shap.Explainer(LGB_opt, Xtrain, feature_names=flist)
shap_values = explainer(Xtest,check_additivity=False)
#%%
shap.plots.beeswarm(shap_values,max_display=9,group_remaining_features=False)

#%% Set up copilot's nifty function to avoid the show() inside the shap plot


save_shap_beeswarm(shap_values, "D:/Project/figures/LGB_beeswarm.png", dpi=300, bbox_inches="tight")
#%% Let's try with the xgb model from above - xgb_model
explainer = shap.Explainer(xgb_model, Xtrain, feature_names=Xtrain_unscaled.columns)
shap_values = explainer(Xtest)
shap.plots.beeswarm(shap_values)
#%% Let's try with the LR model from above, trained on the X3 data
explainer = shap.Explainer(clf3_full, Xtrain, feature_names=Xtrain_unscaled.columns)
shap_values = explainer(Xtest)
shap.plots.beeswarm(shap_values)

#%%
explainer = shap.Explainer(RF3_clf, Xtrain, feature_names=Xtrain_unscaled.columns)
shap_values = explainer(Xtest)
shap.plots.beeswarm(shap_values)
#%%
explainer = shap.Explainer(lgb_clf3, Xtrain, feature_names=Xtrain_unscaled.columns)
shap_values = explainer(Xtest)
shap.plots.beeswarm(shap_values)

#%% Load and test XGB model against the "recent" data, to test predictive power

#Now we're going to load in the "recent" data, so we can see if our models can predict it sensibly...

recent = pd.read_csv("X:/oncology_SJUHBW/Admissions Project/Data Extract/recent_sex_reg_diag.csv")
#%% we're also going to need the csd from another file - yuk!
r2=pd.read_csv("X:/oncology_SJUHBW/Admissions Project/Data Extract/recent_dte_nsh.csv")
#%% try to get r2['csd'] into recent
recent['csd'] = r2['csd']
#%%
recent["csd"] = pd.to_datetime(recent["csd"])
recent.sort_values(axis = 0, inplace=True,by="csd",ascending=True)


#%% Need to process the recent data as above, so it can run through the model(s)
Xr = recent[modelcolsTX]

#%% Let's have a look at the age distribution
plt.hist(Xr[Xr["SubsequentEmergencies"]>0]['BinnedAgeAtCycle'])
plt.title("Age distribution for patients with subsequent admissions")
plt.show()
plt.hist(Xr[Xr["SubsequentEmergencies"]==0]['BinnedAgeAtCycle'])
plt.title("Age distribution for patients without subsequent admissions")
plt.show()


Yr = Xr["SubsequentEmergencies"].astype("int32")
Yr = np.sign(Yr)
Xr.drop(["SubsequentEmergencies"],axis=1,inplace=True)

Xr = Xr.astype("float64")

# OK, now we need to scale using the scaler trained above on the training data
Xrs = scaler.transform(Xr)



#%% Now we can go through in batches of a week at a time?


#%%
Xrs.drop(["Label","csd"],axis=1,inplace=True)
# First we'll run the recent data through a model, generating a new column
#We'll calculate a scale factor for the xgb model, based on the full training data set

#xgbtrainpreds = prior_correct(xgb_model.predict_proba(Xtrain)[:,1])
xgbtrainpreds = prior_correct(XGB_opt.predict_proba(Xtrain)[:,1])
xgb_ratio = np.sum(np.array(xgbtrainpreds))/np.sum(np.array(Ytrain))
print(f"XGB model ratio was {xgb_ratio}")

#rlist = prior_correct(xgb_model.predict_proba(Xrs)[:,1])/xgb_ratio
rlist = prior_correct(XGB_opt.predict_proba(Xrs)[:,1])/xgb_ratio

#%%
# Add back the labels, so we can get both at the same time
Xrs=pd.DataFrame(Xrs)
Xrs["Label"] = Yr
Xrs["csd"] = recent["csd"]
Xrs["predicted"] = rlist
results = []
time_window = 7  #Days
starttime = pd.to_datetime("2024-01-01")
finishtime = pd.to_datetime("2025-12-01")
endtime = starttime + pd.Timedelta(time_window,"days")
while endtime<finishtime:
    predsum = Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["predicted"].sum()
    labelsum = Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["Label"].sum()
    countcases = len(Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["Label"])
    results.append([predsum,labelsum,countcases,labelsum/predsum])
    starttime = endtime
    endtime = starttime + pd.Timedelta(time_window,"days")

Xrs.drop(["Label","csd","predicted"],axis=1,inplace=True)
#%% Plot a graph
pred = np.array([x[0] for x in results])
real = np.array([x[1] for x in results])
cases = np.array([x[2] for x in results])
scalefactor = 1 #np.mean(np.array([x[2] for x in results]))

plt.scatter(pred*scalefactor,real)
plt.xlabel("Predicted emergency admissions")
plt.ylabel("Actual emergency admissions")
plt.savefig("pred_vs_actual_scatter_summed_probs",dpi=300,format='png')

plt.show()

# And a timeline over the two years
plt.plot(pred*scalefactor)
plt.plot(real)
plt.plot(cases/18,alpha=0.5)
plt.legend(['Predicted (summed probabilities)','Actual','Cases / 18'])
plt.xlabel("Week")
plt.ylabel("Emergency admissions")
plt.savefig("projection_summed_probs",dpi=300,format='png')
plt.show()
#%% r2
res = linregress(pred*scalefactor,real)
# res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr, res.intercept_stderr
r2_scipy = res.rvalue**2  # Only valid for simple linear regression
print(r2_scipy)
#%% Let's look at converting the prediction probabilities to a classifier

# Note that the trainpreds have ALREADY been through the prior correct process
for thresh in range(40,50):
    boolres = xgbtrainpreds>(thresh/100.0)
    print(f"For threshold of {thresh/100.0} Number that are positive is {np.sum(boolres)} out of {np.sum(Ytrain)}")
# Looks like a threshold around 0.47 gives about the right number for the training data

#%% Let's try using that for the comparison with the new data...

Xrs=pd.DataFrame(Xrs)
Xrs["Label"] = Yr
Xrs["csd"] = recent["csd"]
Xrs["predicted"] = rlist
Xrs["predicted"] = (Xrs["predicted"]>0.25).astype(int)
results = []
time_window = 7  #Days
starttime = pd.to_datetime("2024-01-01")
finishtime = pd.to_datetime("2025-12-01")
endtime = starttime + pd.Timedelta(time_window,"days")
while endtime<finishtime:
    predsum =Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["predicted"].sum()
    labelsum = Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["Label"].sum()
    countcases = len(Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["Label"])
    results.append([predsum,labelsum,countcases,labelsum/predsum])
    starttime = endtime
    endtime = starttime + pd.Timedelta(time_window,"days")

Xrs.drop(["Label","csd","predicted"],axis=1,inplace=True)

pred = np.array([x[0] for x in results])
real = np.array([x[1] for x in results])
cases = np.array([x[2] for x in results])
scalefactor = 1 #np.mean(np.array([x[2] for x in results]))

plt.plot(pred*scalefactor,real,'.')
print(f'R^2 score is {r2_score(pred*scalefactor,real)}')
m, b = np.polyfit(pred*scalefactor,real, deg=1)
plt.axline(xy1=(40, b+40*m), slope=m,color='red' )
plt.xlabel("Predicted emergency admissions")
plt.ylabel("Actual emergency admissions")
#plt.xlim(40,110)
plt.savefig("pred_vs_actual_scatter_classifier",dpi=300,format='png')

plt.show()

# And a timeline over the two years
plt.plot(pred*scalefactor)
plt.plot(real)
plt.plot(cases/18,alpha=0.5)
plt.legend(['Predicted','Actual','Cases / 18'])
plt.xlabel("Week")
plt.ylabel("Emergency admissions")
plt.savefig("projection_classifier",dpi=300,format='png')
plt.show()

#%%
res = linregress(pred*scalefactor,real)
# res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr, res.intercept_stderr
r2_scipy = res.rvalue**2  # Only valid for simple linear regression
print(r2_scipy)
#%% Let's try it for the LightGBM model

#lgbtrainpreds = prior_correct(lgb_clf3.predict_proba(Xtrain)[:,1])*0.52 +0.04
lgbtrainpreds = prior_correct(LGB_opt.predict_proba(Xtrain)[:,1])*0.52 +0.04
lgb_ratio = 1 #np.sum(np.array(lgbtrainpreds))/np.sum(np.array(Ytrain))
print(f"LGB model ratio was {lgb_ratio}")

#r2list = prior_correct(lgb_clf3.predict_proba(Xrs)[:,1])/lgb_ratio
r2list = prior_correct(LGB_opt.predict_proba(Xrs)[:,1])/lgb_ratio

Xrs["Label"] = Yr
Xrs["csd"] = recent["csd"]
Xrs["predictedLGB"] = r2list
results = []
time_window = 7  #Days
starttime = pd.to_datetime("2024-01-01")
finishtime = pd.to_datetime("2025-12-01")
endtime = starttime + pd.Timedelta(time_window,"days")
while endtime<finishtime:
    predsum = Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["predictedLGB"].sum()
    labelsum = Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["Label"].sum()
    countcases = len(Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["Label"])
    results.append([predsum,labelsum,countcases,labelsum/predsum])
    starttime = endtime
    endtime = starttime + pd.Timedelta(time_window,"days")

Xrs.drop(["Label","csd","predictedLGB"],axis=1,inplace=True)
#%% Plot a graph
pred = np.array([x[0] for x in results])
real = np.array([x[1] for x in results])
cases = np.array([x[2] for x in results])
scalefactor = 1 #np.mean(np.array([x[2] for x in results]))

plt.scatter(pred*scalefactor,real)
m, c = np.polyfit(pred*scalefactor,real, 1)
lr = m*pred*scalefactor +c
plt.plot(pred*scalefactor,lr,color='red', label=f'y = {m:.2f}x + {c:.2f}')
plt.legend()
plt.xlabel("Predicted emergency admissions")
plt.ylabel("Actual emergency admissions")
plt.savefig("D:/Project/figures/pred_vs_actual_scatter_summed_probs",dpi=300,format='png')

plt.show()

# And a timeline over the two years
plt.plot(pred*scalefactor)
plt.plot(real)
plt.plot(cases/16,alpha=0.5)
plt.legend(['Predicted (summed probabilities)','Actual','Cases / 16'])
plt.xlabel("Week")
plt.ylabel("Emergency admissions")
plt.savefig("D:/Project/figures/projection_summed_probs",dpi=300,format='png')
plt.show()
#%% Work on R2 value


res = linregress(pred*scalefactor,real)
# res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr, res.intercept_stderr
r2_scipy = res.rvalue**2  # Only valid for simple linear regression
print(r2_scipy)
#%%
#%% Let's try Logistic regression with the smaller subset
workingfields = Xtrain_unscaled.columns
clf3_full = LogisticRegression(penalty='l1',max_iter=5000,random_state=0,  C = 1.0, solver='liblinear',class_weight='balanced') #,l1_ratio = 1.0
clf3_full.fit(X3train, Y3)
LR3_full_out = clf3_full.predict(X3train)

print(f"The training accuracy is {1-sum(abs(LR3_full_out-Y3))/len(Y3)}")

LR_full_out2 = clf3_full.predict(Xtest)

print(f"The test accuracy is {1-sum(abs(LR_full_out2-Ytest))/len(Ytest)}")

coeffs3 = clf3_full.coef_[0]
importances = []
for n in range(len(coeffs3)):
    if coeffs3[n] > 0: importances.append((abs(coeffs3[n]),workingfields[n]))
importances.sort(reverse = True)                       
print(f"There are {len(importances)} non-zero coefficients")
print(importances[:20])
print(f"There were {sum(LR_full_out2)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print(classification_report(Y3, LR3_full_out))
print(classification_report(Ytest, LR_full_out2)) 
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, LR_full_out2))

#%% Try to get grid search working
LR_clf = LogisticRegression()
param_grid_LR = {"max_iter": [1000,5000],
              "penalty": ["l1","l2"],
              "C": [1.0,10.0],
              "solver": ['liblinear',"saga"],
              "class_weight":["balanced"],
              "random_state":[0]}
LR_opt = GridSearchCV(LR_clf, param_grid_LR, cv=5, scoring='roc_auc', n_jobs=-1)
LR_opt.fit(X3train,Y3)
print("Results from logistic regression:")
print(LR_opt.best_params_)
LR_Opt_test = LR_opt.predict(Xtest)

print(f"The test accuracy is {1-sum(abs(LR_Opt_test-Ytest))/len(Ytest)}")
#%%
coeffsopt3 = LR_opt.best_estimator_.coef_[0]
importances = []
for n in range(len(coeffsopt3)):
    if coeffsopt3[n] > 0: importances.append((abs(coeffsopt3[n]),workingfields[n]))
importances.sort(reverse = True)                       
print(f"There are {len(importances)} non-zero coefficients")
print(importances[:20])
print(f"There were {sum(LR_Opt_test)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print(classification_report(Ytest,LR_Opt_test))
#%% 
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, LR_Opt_test))

#%%
LRout = LR_opt.predict_proba(Xtest)[:,1]
print(classification_report(Ytest, (LRout >= 0.5)))
fpr, tpr, thresh = roc_curve(Ytest, LRout)
plt.plot(fpr, tpr)
plt.title('AUC for LR_opt model with 20:1test  data with 3:1 training= ' + str(roc_auc_score(Ytest,LRout)))
plt.show()

#%% And for a random forest classifier:
    
RF_clf = RandomForestClassifier()
param_grid_RF = {"max_depth": [5,10,20],
              "n_estimators": [200,500,1000],
              "class_weight":["balanced"],
              "random_state":[0]}
RF_opt = GridSearchCV(RF_clf, param_grid_RF, cv=5, scoring='roc_auc', n_jobs=-1)
RF_opt.fit(X3train,Y3)
print("Results from random forest:")
print(RF_opt.best_params_)
RF_Opt_test = RF_opt.predict(Xtest)

print(f"The test accuracy is {1-sum(abs(RF_Opt_test-Ytest))/len(Ytest)}")
#%%

print(f"There were {sum(RF_Opt_test)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print(classification_report(Ytest,RF_Opt_test)) 
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, RF_Opt_test))

RFout = RF_opt.predict_proba(Xtest)[:,1]
print(classification_report(Ytest, (RFout >= 0.5)))
fpr, tpr, thresh = roc_curve(Ytest, RFout)
plt.plot(fpr, tpr)
plt.title('AUC for RF_opt model with 20:1test  data with 3:1 training= ' + str(roc_auc_score(Ytest,RFout)))
plt.show()

#%% Try SVC with the 3:1 data
svc3_clf = SVC() 
param_grid_SVC = {"kernel": ['rbf','linear'],#,'sigmoid'],
              "C": [1.0],#,10.0],
              "class_weight":["balanced"],
              "random_state":[0]}
SVC_opt = GridSearchCV(svc3_clf, param_grid_SVC, cv=5, scoring='roc_auc', n_jobs=-1)
SVC_opt.fit(X3train,Y3)
print("Results from SVC:")
print(SVC_opt.best_params_)
#%%
SVC_Opt_test = SVC_opt.predict(Xtest)

print(f"The test accuracy is {1-sum(abs(SVC_Opt_test-Ytest))/len(Ytest)}")
print(f"There were {sum(SVC_Opt_test)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print(classification_report(Ytest,SVC_Opt_test)) 
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, SVC_Opt_test))

SVCout = SVC_opt.predict_proba(Xtest)[:,1]
print(classification_report(Ytest, (SVCout >= 0.5)))
fpr, tpr, thresh = roc_curve(Ytest, SVCout)
plt.plot(fpr, tpr)
plt.title('AUC for SVC_opt model with 20:1 test  data with 3:1 training= ' + str(roc_auc_score(Ytest,SVCout)))
plt.show()

#%% Try XGBoost grid search
xgb3_clf = xgb.XGBClassifier(booster = 'gbtree', objective="binary:logistic", random_state=42, scale_pos_weight=3)
#xgb_model = xgb.XGBClassifier(booster = 'gblinear', objective="binary:logistic", random_state=42, class_weight='balanced')
param_grid_XGB = {"booster": ['gbtree','gblinear','dart'],
                  "objective":["binary:logistic",'reg:squarederror'],
                  "alpha":[0,1],
                  "lambda":[0,1],
              "max_depth": [4,6,10],
              "scale_pos_weight":[3],
              "random_state":[42]}
XGB_opt = GridSearchCV(xgb3_clf, param_grid_XGB, cv=5, scoring='roc_auc', n_jobs=-1)
XGB_opt.fit(X3train,Y3)
print("Results from XGBoost:")
print(XGB_opt.best_params_)
XGB_Opt_test = XGB_opt.predict(Xtest)

print(f"The test accuracy is {1-sum(abs(XGB_Opt_test-Ytest))/len(Ytest)}")
print(f"There were {sum(XGB_Opt_test)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print(classification_report(Ytest,XGB_Opt_test)) 
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, XGB_Opt_test))
#%%
XGBout = XGB_opt.predict_proba(Xtest)[:,1]
print(classification_report(Ytest, (XGBout >= 0.5)))
fpr, tpr, thresh = roc_curve(Ytest, XGBout)
plt.plot(fpr, tpr)
plt.title('AUC for XGB_opt model with 20:1 test  data with 3:1 training= ' + str(roc_auc_score(Ytest,XGBout)))
plt.show()
#%% Lets try a neural net
nn3_clf = MLPClassifier()
param_grid_NN = {"hidden_layer_sizes": [(50), (317), (317,634), (50, 100), (50,100, 50)],
              "activation": ["relu", "tanh"],
              "max_iter": [500,2000],
              "solver": ["adam"]}
NN_opt = GridSearchCV(nn3_clf, param_grid_NN, cv=5, scoring='roc_auc', n_jobs=-1)
NN_opt.fit(X3train,Y3)
print("Results from MLP:")
print(NN_opt.best_params_)
NN_Opt_test = NN_opt.predict(Xtest)

print(f"The test accuracy is {1-sum(abs(NN_Opt_test-Ytest))/len(Ytest)}")
print(f"There were {sum(NN_Opt_test)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print(classification_report(Ytest,NN_Opt_test)) 
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, NN_Opt_test))
#%%
NNout = NN_opt.predict_proba(Xtest)[:,1]
print(classification_report(Ytest, (NNout >= 0.5)))
fpr, tpr, thresh = roc_curve(Ytest, NNout)
plt.plot(fpr, tpr)
plt.title('AUC for NN_opt model with 20:1 test  data with 3:1 training= ' + str(roc_auc_score(Ytest,NNout)))
plt.show()

#%% As a little aside, we are going to try to save the trained models, so we don't need to train/optimise them again



with open("D:/Project/Trained_Models/NN_opt.pkl",'wb') as f:
    dump(NN_opt,f,protocol = 5)

with open("D:/Project/Trained_Models/XGB_opt.pkl",'wb') as f:
    dump(XGB_opt,f,protocol = 5)
with open("D:/Project/Trained_Models/SVC_opt.pkl",'wb') as f:
    dump(SVC_opt,f,protocol = 5)

with open("D:/Project/Trained_Models/RF_opt.pkl",'wb') as f:
    dump(RF_opt,f,protocol = 5)

with open("D:/Project/Trained_Models/LR_opt.pkl",'wb') as f:
    dump(LR_opt,f,protocol = 5)

#%% ...and try to load one back in, and see if it works:

with open("D:/Project/Trained_Models/NN_opt.pkl",'rb') as f:
    reloadednn_clf = load(f)

relNNout = reloadednn_clf.predict_proba(Xtest)[:,1]
print(classification_report(Ytest, (relNNout >= 0.5)))
fpr, tpr, thresh = roc_curve(Ytest, relNNout)
plt.plot(fpr, tpr)
plt.title('AUC for reloaded NN_opt model with 20:1 test  data with 3:1 training= ' + str(roc_auc_score(Ytest,NNout)))
plt.show()

#%% OK, aside over, let's try to use grid search for LightGBM

lgb3_clf = lgb.LGBMClassifier()

param_grid_LGB = {"num_leaves": [30,50,80,120],
              "min_data_in_leaf": [20,50,100,500,2000],
              "num_iterations": [100,500],
              "lambda_l1": [0,1],
              "lambda_l2": [0,1],
              "verbose":[-1]}
LGB_opt = GridSearchCV(lgb3_clf, param_grid_LGB, cv=5, scoring='roc_auc', n_jobs=-1)
LGB_opt.fit(X3train,Y3)
print("Results from LightGBM:")
print(LGB_opt.best_params_)
LGB_Opt_test = LGB_opt.predict(Xtest)

print(f"The test accuracy is {1-sum(abs(LGB_Opt_test-Ytest))/len(Ytest)}")
print(f"There were {sum(LGB_Opt_test)} predicted and {sum(Ytest)} actual admissions in the test set.")                             
print(classification_report(Ytest,LGB_Opt_test)) 
print("Confusion matrix for test data")
print(confusion_matrix(Ytest, LGB_Opt_test))

#%% With probabilities, and let's see what prior correct does
threshold = 0.07
LGBout = prior_correct(LGB_opt.predict_proba(Xtest)[:,1])
print(classification_report(Ytest, (LGBout >= threshold)))
print(confusion_matrix(Ytest, (LGBout >= threshold)))
# fpr, tpr, thresh = roc_curve(Ytest, LGBout)
# plt.plot(fpr, tpr)
# plt.title('AUC for LGB_opt model with 20:1 test  data with 3:1 training corrected = ' + str(roc_auc_score(Ytest,LGBout)))
# plt.show()
#%% Save model...

with open("D:/Project/Trained_Models/LGB_opt.pkl",'wb') as f:
    dump(LGB_opt,f,protocol = 5)
    
#%% Try to run cross-validation on the full dataset, using the "optimal" models we've just got from grid searching...

###########################################################
    
    CVresults_df = pd.DataFrame(CVresults_list,columns=['precision','recall','f1','support','accuracy','balanced acc'])
    return CVresults_df
###########################################################
#%% LightGBM
LGB_CV_res = Do_CV_Results(LGB_opt,Xtrain_unscaled,Ytrain)
#%% OK, that worked, let's do the same for the others

NN_res = Do_CV_Results(NN_opt,Xtrain_unscaled,Ytrain)
print(NN_res)
print("means\n",NN_res.mean())
print("SDs\n",NN_res.std())

#%% Random forest
RF_res = Do_CV_Results(RF_opt,Xtrain_unscaled,Ytrain)
print("Results for Random Forest")
print(RF_res)
print("means\n",RF_res.mean())
print("SDs\n",RF_res.std())
#%% XGBoost
XGB_res = Do_CV_Results(XGB_opt,Xtrain_unscaled,Ytrain)
print("Results for XGBoost")
print(XGB_res)
print("means\n",XGB_res.mean())
print("SDs\n",XGB_res.std())
#%% SVC
SVC_res = Do_CV_Results(SVC_opt,Xtrain_unscaled,Ytrain)
print("Results for SVC")
print(SVC_res)
print("means\n",SVC_res.mean())
print("SDs\n",SVC_res.std())
#%% LR
LR_res = Do_CV_Results(LR_opt,Xtrain_unscaled,Ytrain)
print("Results for LR")
print(LR_res)
print("means\n",LR_res.mean())
print("SDs\n",LR_res.std())

#%% OK, SVC and LR runs IMPOSSIBLY slowly with such a big dataset, so we'll have to cut it down
# Maybe we can use the 20% unscaled test set?
# Xtest_unscaled, Ytest
SVCs_clf = SVC(C = 1.0, class_weight='balanced', kernel='linear', random_state= 0)
LRs_clf = LogisticRegression(C= 1.0, class_weight= 'balanced', max_iter= 1000, penalty= 'l1', random_state= 0, solver= 'liblinear')

SVC_res = Do_CV_Results(SVCs_clf,Xtest_unscaled,Ytest)
print("Results for SVC using only test data")
print(SVC_res)
print("means\n",SVC_res.mean())
print("SDs\n",SVC_res.std())

# LR_res = Do_CV_Results(LRs_clf,Xtrain_unscaled,Ytrain)
# print("Results for LR")
# print(LR_res)
# print("means\n",LR_res.mean())
# print("SDs\n",LR_res.std())
#%% Reload "opt" models, use them to generate forecasts for the "recent" data
# Then we'll find thresholds based on the training data to get the right number of admissions, 
# and try applying those to the recent data

with open("D:/Project/Trained_Models/NN_opt.pkl",'rb') as f:
    NN_opt = load(f).best_estimator_

with open("D:/Project/Trained_Models/XGB_opt.pkl",'rb') as f:
    XGB_opt = load(f).best_estimator_
with open("D:/Project/Trained_Models/SVC_opt.pkl",'rb') as f:
    SVC_opt = load(f).best_estimator_

with open("D:/Project/Trained_Models/RF_opt.pkl",'rb') as f:
    RF_opt = load(f).best_estimator_

with open("D:/Project/Trained_Models/LR_opt.pkl",'rb') as f:
    LR_opt = load(f).best_estimator_

with open("D:/Project/Trained_Models/LGB_opt.pkl",'rb') as f:
    LGB_opt = load(f).best_estimator_

#%% Let's m ake a function to take a set of prediction probabilities and a "true" number of positives, 
# and use a binary search to find a threshold that yields that number

    
LGB_tr = prior_correct(LGB_opt.predict_proba(Xtrain)[:,1])
print(classification_report(Ytrain, (LGB_tr >= 0.5)))
print(confusion_matrix(Ytrain, (LGB_tr >= 0.5)))
LGB_thresh = BinSearchThreshold(LGB_tr,np.sum(Ytrain),min=0.0,max=1.0)
print(f"The threshold foudn for the LGB model was {LGB_thresh}")
print(classification_report(Ytrain, (LGB_tr >= LGB_thresh)))
print(confusion_matrix(Ytrain, (LGB_tr >= LGB_thresh)))

# Now we predict for the "recent" dataset Xrs using this threshold
LGB_re = prior_correct(LGB_opt.predict_proba(Xrs)[:,1])>LGB_thresh

Xrs["LGB"]= LGB_re.astype("int32")
preds_df = pd.DataFrame(Xrs["LGB"],columns=["LGB"])
Xrs.drop(["LGB"],inplace=True,axis=1)
#%% Now the same for the other models

XGB_thresh = BinSearchThreshold(prior_correct(XGB_opt.predict_proba(Xtrain)[:,1]),np.sum(Ytrain),min=0.0,max=1.0)
print(f"The threshold found for the XGB model was {XGB_thresh}")

# Now we predict for the "recent" dataset Xrs using this threshold
preds_df["XGB"]= prior_correct(XGB_opt.predict_proba(Xrs)[:,1])>XGB_thresh
n=np.sum(preds_df["XGB"])
print(f"There were {n} positives returned for the XGB model")

#%% NN/SVC/RF/LR
NN_thresh = BinSearchThreshold(prior_correct(NN_opt.predict_proba(Xtrain)[:,1]),np.sum(Ytrain),min=0.0,max=1.0)
print(f"The threshold found for the NN model was {NN_thresh}")
preds_df["NN"]= prior_correct(NN_opt.predict_proba(Xrs)[:,1])>NN_thresh

RF_thresh = BinSearchThreshold(prior_correct(RF_opt.predict_proba(Xtrain)[:,1]),np.sum(Ytrain),min=0.0,max=1.0)
preds_df["RF"]= prior_correct(RF_opt.predict_proba(Xrs)[:,1])>RF_thresh

LR_thresh = BinSearchThreshold(prior_correct(LR_opt.predict_proba(Xtrain)[:,1]),np.sum(Ytrain),min=0.0,max=1.0)
preds_df["LR"]= prior_correct(LR_opt.predict_proba(Xrs)[:,1])>XGB_thresh
#%% Now we'll make a column that sums those 5 model predictions, and look at using those which score 3 or more
preds_df = preds_df.astype("int32")
preds_df["Sum"] = preds_df["LGB"]+preds_df["XGB"]+preds_df["NN"]+preds_df["RF"]+preds_df["LR"]
print(classification_report(Yr, (preds_df["Sum"] >= 3)))
#%%
model = "Sum"
Xrs["Label"] = Yr
Xrs["csd"] = recent["csd"]
Xrs["predicted"] = (preds_df[model]>0).astype("int32")
results = []
time_window = 7  #Days
starttime = pd.to_datetime("2024-01-01")
finishtime = pd.to_datetime("2025-12-01")
endtime = starttime + pd.Timedelta(time_window,"days")
while endtime<finishtime:
    predsum = Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["predicted"].sum()
    labelsum = Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["Label"].sum()
    countcases = len(Xrs[(Xrs["csd"]>= starttime) & (Xrs["csd"]<endtime)]["Label"])
    results.append([predsum,labelsum,countcases,labelsum/predsum])
    starttime = endtime
    endtime = starttime + pd.Timedelta(time_window,"days")

Xrs.drop(["Label","csd","predicted"],axis=1,inplace=True)


# Plot a graph
pred = np.array([x[0] for x in results])
real = np.array([x[1] for x in results])
cases = np.array([x[2] for x in results])
scalefactor = 1 #np.mean(np.array([x[2] for x in results]))

plt.scatter(pred*scalefactor,real)
m, c = np.polyfit(pred*scalefactor,real, 1)
lr = m*pred*scalefactor +c
plt.plot(pred*scalefactor,lr,color='red', label=f'y = {m:.2f}x + {c:.2f}')
plt.legend()
plt.xlabel("Predicted emergency admissions")
plt.ylabel("Actual emergency admissions")
plt.savefig("pred_vs_actual_scatter_summed_probs",dpi=300,format='png')

plt.show()
res = linregress(pred*scalefactor,real)
# res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr, res.intercept_stderr
r2_scipy = res.rvalue**2  # Only valid for simple linear regression
print(r2_scipy)

# And a timeline over the two years
# plt.plot(pred*scalefactor)
# plt.plot(real)
# plt.plot(cases/16,alpha=0.5)
# plt.legend(['Predicted (summed probabilities)','Actual','Cases / 16'])
# plt.xlabel("Week")
# plt.ylabel("Emergency admissions")
# plt.savefig("projection_summed_probs",dpi=300,format='png')
# plt.show()
#%% confusion matrices
print(f"There are a total of {np.sum(Yr)} true positives")
print(f"Confusion matrix for LR:\n{confusion_matrix(Yr, preds_df['LR'])}")
print(f"Confusion matrix for RF:\n{confusion_matrix(Yr, preds_df['RF'])}")
print(f"Confusion matrix for NN:\n{confusion_matrix(Yr, preds_df['NN'])}")
print(f"Confusion matrix for XGB:\n{confusion_matrix(Yr, preds_df['XGB'])}")
print(f"Confusion matrix for LGB:\n{confusion_matrix(Yr, preds_df['LGB'])}")

#%%
LGBtest_probs = LGB_opt.predict_proba(Xtest)
#results,numbers = CalibCurve(LGBtest_probs,Ytest,"LGB_3:1_training")
LGB_test=LGBtest_probs.tolist()
LGB_test1 = [x[1] for x in LGB_test]

fpr, tpr, thresh = roc_curve(Ytest, LGB_test1)
plt.plot(fpr, tpr)
#plt.title('AUC for test LightGBM data with 3:1 training= ' + str(roc_auc_score(Ytest,LGB_test1)))
plt.savefig("d:/Project/figures/LGB_auroc.png",dpi=300,format="png")
plt.show()

#%% Calbration curve corrected

results,numbers = CalibCurveUntitled(prior_correct(LGBtest_probs),Ytest,"LGB_3")

#%% Check that our predictions make some sort of sense - scatter plots look more or less random...
# Applying our LGB model to the "recent" dataset gives results that aren't as good as for the main set, but not horrible

LGB_out = LGB_opt.predict_proba(Xrs)

print(classification_report(Yr, (LGB_out[:,1]>0.7)))
print(confusion_matrix(Yr, (LGB_out[:,1]>0.7)))
fpr, tpr, thresh = roc_curve(Yr, LGB_out[:,1])
plt.plot(fpr, tpr)
plt.title('AUC for LGB_opt model with 20:1 test  data with 3:1 training= ' + str(roc_auc_score(Yr,LGB_out[:,1])))
plt.show()
