"""Train-once sklearn adapter for the faithful demo, defects preserved."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.utils.validation import validate_data,check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from .reference import trace_reference,reference_prediction


class FaithfulIFUTSVMID(ClassifierMixin,BaseEstimator):
    """Faithful baseline with sorted classes mapped to -1,+1.

    The original incorrect label swap is retained. Nearly balanced training
    counts can raise, just as in trace_reference. No scaling is implicit;
    compare raw and StandardScaler pipeline configurations separately.
    """
    def __init__(self,C=1.0,mu=1.0,eps=1.0):
        self.C=C
        self.mu=mu
        self.eps=eps

    def fit(self,X,y):
        X,y=validate_data(self,X,y,dtype=np.float64)
        check_classification_targets(y)
        self.classes_=np.unique(y)
        if len(self.classes_)!=2:
            raise ValueError('faithful estimator requires exactly two classes')
        encoded=np.where(y==self.classes_[1],1.,-1.)
        train=np.column_stack((X,encoded))
        # A single training row supplies the required trace API test argument;
        # no evaluation data enter fitting. Prediction then reuses the planes.
        self.trace_=trace_reference(train,train[:1],self.C,self.mu,self.eps)
        return self

    def _prediction(self,X):
        check_is_fitted(self,'trace_')
        X=validate_data(self,X,reset=False,dtype=np.float64)
        t=self.trace_
        return reference_prediction(X,t['C'],t['w1'],t['w2'],self.mu)

    def predict(self,X):
        encoded=self._prediction(X)[3]
        return np.where(encoded==1,self.classes_[1],self.classes_[0])

    def decision_function(self,X):
        _,a,b,_=self._prediction(X)
        return np.abs(b)-np.abs(a)
