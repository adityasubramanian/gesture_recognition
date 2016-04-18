import numpy as np
import pickle
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from crf import LinearCRF, LinearCRFEnsemble
from svmhmm import SVMHMMCRF
from sklearn.metrics import accuracy_score
from utils import *
from utilities import * 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm.classes import LinearSVC
from sklearn import cross_validation
from sklearn.feature_selection import RFECV

lables = ['1', '2', '3', '4', '5', '6']
label_meanings = {
	'SGD Classifier': '1', 
	'Logistic Regression': '2', 
	'Linear Support Vector Classifier': '3', 
	'Gaussian Naive Bayes': '4', 
	'KNN (N = 5)': '5', 
	'Random Forest': '6'
}

def SVM_feature_extraction(X_train, y_train, X_test):
    """
    SVM Classifier
    """ 
    clf = svm.LinearSVC() # Initializing a svm Linear SVC
    clf.fit(X_train, y_train) # Fitting the svc on X and y_train 
    X_train_t = clf.decision_function(X_train)  # Returning train and test for the X_ values after decision_function() has been applied. 
    X_test_t = clf.decision_function(X_test)
    return (X_train_t,X_test_t)

# Use SVD analysis. 

def fit_clf_kfold(clf,Xs,ys,flatten=True,n_folds=5, add_last_action=False): 
    """
    X: an array of X, one for each person
    y: an array of array of labels, one for each person
    flatten: set to True for classifiers that don't take the structure into account.
    add_last_action: add last action as a feature?
    
    return:
    (y_gold, y_predict) for each fold
    """
    
    result = []
    
    kf = cross_validation.KFold(len(Xs), n_folds=n_folds, shuffle=True, random_state=1)
    for i,(train_index, test_index) in enumerate(kf):
        print "fold %d" % i
        X_train = [Xs[u] for u in train_index]
        y_train = [ys[u] for u in train_index]
        X_test = [Xs[u] for u in test_index]
        y_test = [ys[u] for u in test_index]
        if flatten:
            print "flattening"
            X_train, y_train = flatten_data(X_train, y_train)
            X_test, y_test = flatten_data(X_test, y_test)
            if not add_last_action:
                clf.fit(X_train,y_train)
                y_predict = clf.predict(X_test)
            else:
                onehot, X_train_new = get_last_action_feature(X_train,y_train)
                clf.fit(X_train_new,y_train)
                y_predict = predict_with_last_action(clf,X_test,onehot)
            y_gold = y_test
        else:
            clf.batch_fit(X_train,y_train)
            y_predict = clf.batch_predict(X_test)
            y_predict = np.concatenate(y_predict)
            y_gold = np.concatenate(y_test)
        
        result.append((y_gold,y_predict))
    return result

def get_diff_features(X):
    X_diff = np.diff(X, n=1, axis=0)
    Xnew = np.zeros(X.shape)
    Xnew[1:,:] = X_diff
    return Xnew
def run_clfs_on_data(classifiers, Xs, ys, add_last_action = False):
    results = {}
    for name, clf in classifiers.iteritems():
        print "Currently Running: %s" % name
        clf_results = fit_clf_kfold(clf['clf'], Xs, ys, flatten=not clf['structured'], add_last_action=add_last_action)
        # with feature selection:
		#clf_results = fit_clf_kfold(clf['clf'], [X[:,select_features] for X in X_pers_all], y_pers_all,flatten=not clf['structured'])
        results[name] = clf_results
    return results

def plot_most_important_features(clf, label_names, feature_names, n=10, best=True, absolut=True):
    if absolut:
        ranked_features = np.argsort(np.abs(clf.coef_), axis=None)
    else:
        ranked_features = np.argsort(clf.coef_, axis=None)
    if best:
        ranked_features = ranked_features[::-1] 
    for i, fweights_idx in enumerate(ranked_features[:n]):
            label_idx,feature_idx = np.unravel_index(fweights_idx, clf.coef_.shape)
            print "%d. f: %s\t\t c: %s\t value: %f" % (i, feature_names[feature_idx], label_names[label_idx], clf.coef_[(label_idx,feature_idx)])

def unflatten_per_person(X_all,y_all,persons_all):
    """
        X: n_samples, n_features.  The full feature matrix.
        y: label for each row in X person: person label for each row in X
        returns: (X_person, y_person) X_person: n_persons array of X and y that apply to this person.
    """
    Xtotal, y_total, Xperson, y_person = [], [], [], [] 
    last_person = persons_all[0]
    for row,y,person in zip(X_all,y_all,persons_all):
        if person != last_person:
            Xtotal.append(Xperson)
            y_total.append(y_person)
            Xperson, y_person = [], []            
        Xperson.append(row)
        y_person.append(y)
        
        last_person = person
        
    Xtotal.append(Xperson)
    y_total.append(y_person)
    
    return ([np.array(x) for x in Xtotal], [np.array(y) for y in y_total])

def flatten_data(X,y):
    return np.concatenate(X), np.concatenate(y)

def get_last_action_feature(X,ys):
    """
    Convert the labels in ys into a one-hot representation.
    The result will be shifted so that each row represents the last action.
    The first row everything will be 0.
    """
    onehot = OneHotEncoder()
    onehot.fit([[y] for y in ys])
    
    actions = onehot.transform([[y] for y in ys])
    actions = np.asarray(actions.todense())
    #the first row is all zeros, because there is no prior action:
    last_action = np.zeros(actions.shape)
    last_action[1:,:] = actions[:-1,:]

    return onehot,np.concatenate([X,last_action],axis=1)
    
def predict_with_last_action(clf, X, onehot):
    """
        Predict one at a time and always add last action to 
        the next prediction using the onehot encoder.
    """
    lasty = np.zeros(6) #TODO: remove harding of 6
    y_predict = []
    for x in X:
        x_last_action = np.concatenate([x,lasty])
        y = clf.predict([x_last_action])
        y_predict.append(y[0])
        lasty = np.array(onehot.transform([y]).todense()).flatten()
        
    return y_predict
    
def num_label_changes(y):
    """
        For a label sequence this function calculates the number of times the label changes.
        e.g. num_label_changes([1,1,1,2,2,2,3,3]) = 2
    """
    num_changes = 0
    for y, y_next in zip(y, y[1:]):
        if y != y_next:
            num_changes += 1
    return num_changes

def label_smoothness(y_predict):
    """
        Number of label transitions over number of labels.
        
        The smaller the smoother it is.
    """
    n = len(y_predict)
    num_changes_predict = num_label_changes(y_predict)
    return num_changes_predict / float(n)

if __name__ == '__main__':
    print "Loading DATA"

    print "Training Data.. (X) "
    X_train = np.loadtxt('/home/aditya/Documents/UIUC/spring_2016/stat427/project/uci_har_dataset/train/X_train.txt')

    print "Training Data.. (Y) "
    y_train = np.loadtxt('/home/aditya/Documents/UIUC/spring_2016/stat427/project/uci_har_dataset/train/y_train.txt', dtype=np.int)

    print "Training Data.. (Persons)"
    persons_train = np.loadtxt('/home/aditya/Documents/UIUC/spring_2016/stat427/project/uci_har_dataset/train/subject_train.txt', dtype=np.int)

	#persons_train = np.loadtxt('/home/aditya/Documents/UIUC/spring_2016/stat427/project/train.txt', dtype=np.int)
	

    print "Testing Data.. (X) "
    X_test = np.loadtxt('/home/aditya/Documents/UIUC/spring_2016/stat427/project/uci_har_dataset/test/X_test.txt')

    print "Testing Data.. (Y) "
    y_test = np.loadtxt('/home/aditya/Documents/UIUC/spring_2016/stat427/project/uci_har_dataset/test/y_test.txt', dtype=np.int)

    print "Testing Data.. (Persons) "
    persons_test = np.loadtxt('/home/aditya/Documents/UIUC/spring_2016/stat427/project/uci_har_dataset/test/subject_test.txt', dtype=np.int)

    #persons_test = np.loadtxt('/home/aditya/Documents/UIUC/spring_2016/stat427/project/test.txt', dtype=np.int)

    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    
    feature_names = [x.split(' ')[1] for x in open('/home/aditya/Documents/UIUC/spring_2016/stat427/project/uci_har_dataset/features.txt').read().split('\n') if len(x) > 0]
    
    #SVM-HMM dumping
    #dump_svmlight_file(X_test,y_test,"/Users/tdomhan/Downloads/svm_hmm/activity-data/Xtest.data",zero_based=False,query_id=persons_test)
    #dump_svmlight_file(X_train,y_train,"/Users/tdomhan/Downloads/svm_hmm/activity-data/Xtrain.data",zero_based=False,query_id=persons_train)
    #laod the SVM-HMM predictions
    #y_predict = np.loadtxt("/Users/tdomhan/Downloads/svm_hmm/classfy.tag", dtype=int)
    
    feature_names = [x.split(' ')[1] for x in open('/home/aditya/Documents/UIUC/spring_2016/stat427/project/uci_har_dataset/features.txt').read().split('\n') if len(x) > 0]
    
    """ Split by person: """
    X_train_pers, y_train_pers = unflatten_per_person(X_train, y_train, persons_train)
    X_test_pers, y_test_pers = unflatten_per_person(X_test, y_test, persons_test)
    
    X_pers_all = []
    X_pers_all.extend(X_train_pers)
    X_pers_all.extend(X_test_pers)
    y_pers_all = []
    y_pers_all.extend(y_train_pers)
    y_pers_all.extend(y_test_pers)
    
    print "Status: Training Classifiers"
    
    ensemble_classifiers = {
                                "Linear Support Vector Classifier": {'clf': LinearSVC(), 'structured': False},
                                "Logistic Regression": {'clf': LogisticRegression(), 'structured': False},
                                "SGDClassifier":{'clf': SGDClassifier(),'structured':False},
                                }
    
    crf_ensemble = LinearCRFEnsemble(ensemble_classifiers, addone=True, regularization=None, lmbd=0.01, sigma=100, transition_weighting=True)
    
    classifiers = {
                  "SGDClassifier":{'clf': SGDClassifier(),'structured':False},
                   "Logistic Regression": {'clf': LogisticRegression(), 'structured': False},
                   "linear Support Vector Classifier": {'clf': LinearSVC(), 'structured': False},
                   "Gaussian Naive Bayes": {'clf': GaussianNB(), 'structured': False},
                   "KNN (weights: uniform, neighbors=5)": {'clf': KNeighborsClassifier(), 'structured': False},
                   "RandomForest": {'clf': RandomForestClassifier(), 'structured': False},
                   }

    results = run_clfs_on_data(classifiers, X_pers_all, y_pers_all)
    
    results_last_action = run_clfs_on_data(classifiers, X_pers_all, y_pers_all, add_last_action=True)
    
    for clf_name in results:
        clf_results = results[clf_name]
        accuracies = np.array([accuracy_score(gold, predict) for gold, predict in clf_results])
        print accuracies
        print "%s accuracy: %f +- %f" % (clf_name, accuracies.mean(), accuracies.std())
        smoothness_predict = np.array([label_smoothness(predict) for gold, predict in clf_results])
        print "%s smoothness: %f +- %f" % (clf_name, smoothness_predict.mean(), smoothness_predict.std())
        smoothness_gold = np.array([label_smoothness(gold) for gold, predict in clf_results])
        print "smoothess(gold): %f +- %f" % (smoothness_gold.mean(), smoothness_gold.std())
        
        y_all_gold = np.concatenate(zip(*clf_results)[0])
        y_all_predict = np.concatenate(zip(*clf_results)[1])

        #print classification_report(y_all_gold, y_all_predict, target_names = labels)
        print confusion_matrix_report(y_all_gold, y_all_predict, labels)
        print confusion_matrix(y_all_gold, y_all_predict)
        
    crf_classifiers =  {
                        "CRF": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=100, transition_weighting=False),
                            'structured': True},
                        "CRF transition weights": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=100, transition_weighting=True),
                            'structured': True},
                        }
    
    crf_unregularized_classifiers =  {
                        "CRF": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization=None, lmbd=0.01, sigma=10, transition_weighting=False),
                            'structured': True},
                        "CRF transition weights": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization=None, lmbd=0.01, sigma=10, transition_weighting=True),
                            'structured': True},
                        }
    
    crf_classifiers_l2_best = {
                   "CRF (sigma=1)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=1, transition_weighting=False),
                            'structured': True},
                   "CRF (sigma=10)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=10, transition_weighting=False),
                            'structured': True},
                    "CRF (sigma=100)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=100, transition_weighting=False),
                            'structured': True},
                    "CRF (sigma=1000)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=1000, transition_weighting=False),
                            'structured': True},
                    "CRF (sigma=.1)": {'clf': LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization="l2", lmbd=0.01, sigma=0.1, transition_weighting=False),
                            'structured': True},
                   }
    
    results_feature_selection = run_clfs_on_data(classifiers, [RFECV.transform(X) for X in X_pers_all], y_pers_all)
    

    
    clf_results = fit_clf_kfold(clf, X_pers_all, y_pers_all,flatten=False)
        
    clf = svm.SVC()
    clf = LogisticRegression()
    clf = LogisticRegression(penalty='l1',C=100)
    clf = SGDClassifier()
    clf = GaussianNB()
    clf = DecisionTreeClassifier()
    clf = GradientBoostingClassifier()
    
    clf = RandomForestClassifier()
    
    
#    diff features
    diff_scaler = Scaler()
    X_train_diff = get_diff_features(X_train)
   
    diff_scaler.fit(X_train_diff)
    X_train_diff = diff_scaler.transform(X_train_diff)
    X_train_diff = np.concatenate([X_train, X_train_diff],axis=1)
   
    X_test_diff  = get_diff_features(X_test)
    X_test_diff = diff_scaler.transform(X_test_diff)
    X_test_diff  = np.concatenate([X_test, X_test_diff],axis=1)


    onehot,X_train_last_action = get_last_action_feature(X_train, y_train)
    clf = LinearSVC() # things get worse when adding the last action to a linear SVM classifier
    clf = svm.SVC(kernel='poly') # same is true for the poly kernel svm
    clf = svm.SVC()#and also with a rbf kernel
    clf = svm.SVC() 
    clf = RandomForestClassifier()
    clf.fit(X_train_last_action, y_train)
       
    y_predict = predict_with_last_action(clf, X_test, onehot)


    clf = LinearCRF(feature_names=feature_names, label_names=labels, addone=True, regularization=None, lmbd=0.01, sigma=100, transition_weighting=False)

#    a single chain:
    clf.fit(X_train, y_train, X_test, y_test)
    y_predict = clf.predict(X_test)
 #   one chain per person

    clf.batch_fit(X_train_pers, y_train_pers, X_test_pers, y_test_pers)
    y_predict = np.concatenate(clf.batch_predict(X_test_pers))

    clf = LinearCRF(sigma=10)
    X_train_svm, X_test_svm = SVM_feature_extraction(X_train, y_train, X_test)
    clf = LinearCRF(addone=True,sigma=100)
    clf.fit(X_train, y_train, X_test, y_test)

    clf.fit(X_train_svm, y_train, X_test_svm, y_test)
    y_predict = clf.predict(X_test_svm)

    clf.fit(X_train, y_train)  

    print "predicting test data"

    y_predict = clf.predict(X_test)

    print classification_report(y_test, y_predict, target_names = labels)

    print confusion_matrix_report(y_test, y_predict, labels)
    print label_meanings
    # measure the transitions we get right:
print "done"
    
    
    
