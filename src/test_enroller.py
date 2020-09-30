import os
from six.moves import cPickle as pickle

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class EnrollerEvaluator:
    """
    A class to evaluate the quality of the embeddings extractor. From a set of
      image embeddings, a classification model is built, and the accuracy of
      this model is calculated.

    Attributes
    ----------
    embeddings : list<list<int>>
        list of embeddings for the classifier to enroll
    labels : list<int>
        list of the dog breed labels respective to the embeddings
    classes : list<str>
        list of dog breeds that were enrolled to the classifier
    test_embeddings : list<list<int>>
        list of embeddings of the test set that will evaluate the classifier
    test_labels : list<str>
        list of the dog breed labels respective to the embeddings that will
            evaluate the classifier
    classifier : sklearn.KNeighborsClassifier
        a classifier for predicting dog breeds from embeddings

    Methods
    -------
    create_classifier()
        Creates a new dog breed classifier from the embeddings and labels to be
            enroller.

    predict_label(dists, indexes)
        Predicts the dog breed of an input's embeddings regarding its nearest
            existing trained samples. Currently, based on "JÚNIOR, Pedro R. 
            Mendes et al. Nearest neighbors distance ratio open-set classifier.
            Machine Learning, v. 106, n. 3, p. 359-386, 2017."

    evaluate()
        Calculates the accuracy of the classifier on the testing set
    """

    def __init__(self, initial_enroll_path, new_enroll_path, new_enroll_test_path, is_train_enrolled, is_test_enrolled):
        '''
        
        '''

        assert is_train_enrolled or is_test_enrolled, \
            '\'train/\' set or \'recognition/enroll/\' set must be enrolled!'
        
        with open(initial_enroll_path, 'rb') as f:
            initial_enroll = pickle.load(f)
        with open(new_enroll_path, 'rb') as f:
            new_enroll = pickle.load(f)
        with open(new_enroll_test_path, 'rb') as f:
            new_enroll_test = pickle.load(f)
            
        if is_train_enrolled and is_test_enrolled:
            print('case 1')
            self.embeddings = initial_enroll['embeddings']
            self.labels = initial_enroll['labels']
            self.classes = initial_enroll['classes']
            
            new_enroll['labels'] = [l + len(self.classes)
                for l in new_enroll['labels']]
            new_enroll_test['labels'] = [l + len(self.classes)
                for l in new_enroll_test['labels']]
            
            self.embeddings += new_enroll['embeddings']
            self.labels += new_enroll['labels']
            self.classes += new_enroll['classes']
              
        elif is_train_enrolled:
            print('case 2')
            self.embeddings = initial_enroll['embeddings']
            self.labels = initial_enroll['labels']
            self.classes = initial_enroll['classes']
            
            new_enroll_test['labels'] = [-1 for l in new_enroll_test['labels']]
            
        elif is_test_enrolled:
            print('case 3')
            self.embeddings = new_enroll['embeddings']
            self.labels = new_enroll['labels']
            self.classes = new_enroll['classes']
        
        self.test_embeddings = new_enroll_test['embeddings']
        self.test_labels = new_enroll_test['labels']
            
        self.create_classifier()
        
        # self.evaluate()
        
    def create_classifier(self):
        self.classifier = KNeighborsClassifier(n_neighbors=5,
            weights='distance', metric='euclidean')
        self.classifier.fit(self.embeddings, self.labels)
        
    def predict_label(self, dists, indexes, T):
        t = 0
        dist_t, index_t = dists[t], indexes[t]
        
        u = t
        for i_index, index in enumerate(indexes[t:]):
            if index != index_t:
                u = i_index
                dist_u = dists[u]
                break
        if u == t:
            return index_t
        
        R = dist_t / dist_u
        pred = index_t if R <= T else -1
        
        return pred
        
    def evaluate(self):
        test_dists, test_indexes = self.classifier.kneighbors(self.test_embeddings)
        test_indexes = np.vectorize(lambda x: self.labels[x])(test_indexes)
        
        test_dists_indexes = list(zip(test_dists, test_indexes))
        
        accs = []
        for T in np.arange(0.5, 1, 0.05):
            T = round(T, 2)
            test_preds = np.array([self.predict_label(dists, indexes, T)
                for dists, indexes in test_dists_indexes])
            acc = np.sum(test_preds == self.test_labels) / len(self.test_labels)
            
            accs.append(acc)
            print('T = %s, acc = %s' % (T, acc))
            
        accs = np.array(accs)
            
        return accs
            
if __name__ == '__main__':
    initial_enroll_path = os.path.join('..', 'models', 'initial_enroll.pkl')
    new_enroll_path = os.path.join('..', 'models', 'new_enroll.pkl')
    new_enroll_test_path = os.path.join('..', 'models', 'new_enroll_test.pkl')

    enroller_evaluator_1 = EnrollerEvaluator(initial_enroll_path, new_enroll_path,
        new_enroll_test_path, is_train_enrolled=True, is_test_enrolled=True)
    accs_1 = enroller_evaluator_1.evaluate()
    
    enroller_evaluator_2 = EnrollerEvaluator(initial_enroll_path, new_enroll_path,
        new_enroll_test_path, is_train_enrolled=True, is_test_enrolled=False)
    accs_2 = enroller_evaluator_2.evaluate()
    
    enroller_evaluator_3 = EnrollerEvaluator(initial_enroll_path, new_enroll_path,
        new_enroll_test_path, is_train_enrolled=False, is_test_enrolled=True)
    accs_3 = enroller_evaluator_3.evaluate()
