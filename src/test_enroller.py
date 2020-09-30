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
            Machine Learning, v. 106, n. 3, p. 359-386, 2017.".

    evaluate()
        Calculates the accuracy of the classifier on the testing set.
    """

    def __init__(self, initial_enroll_path, new_enroll_path, new_enroll_test_path, is_train_enrolled, is_test_enrolled):
        '''
        Constructs all the attributes for the enroller evaluation object.

        Parameters
        ----------
        initial_enroll_path : str
            path to the file containing the embeddings and labels of the
                training set samples (initial 100 dog breeds)
        new_enroll_path : str
            path to the file containing the embeddings and labels of the
                unseen test set samples to be enrolled (external 20 dog breeds)
        new_enroll_test_path : str
            path to the file containing the embeddings and labels of the
                unseen test set samples to be tested (same external 20 dog
                breeds)
        is_train_enrolled : bool
            Defines if the training set samples will be enrolled to the
                classifier
        is_test_enrolled : bool
            Defines if the unseen test set samples will be enrolled to the
                classifier
        '''

        '''
        Three evaluations are conducted:
        - Evaluation 1: Both training ('dogs/train/') and unseen test
            ('dogs/recognition/enroll/') embeddings sets are enrolled. Therefore,
            the labels of the test set ('dogs/recognition/test/') are the same of
            the unseen test; ideally, no unknown label should be predicted;
        - Evaluation 2: Only the training set ('dogs/train/') is enrolled to the
            classifier. Therefore, when evaluating the test set
            ('dogs/recognition/test/'), all predicted labels should refer to
            unknown dog breed. This is the open-space evaluation;
        - Evaluation 3: Only the unseen test set ('dogs/recognition/enroll/') is
            enrolled. In this case, the classifier will only contain the external
            20 dog breeds, which are the same as the test set
            ('dogs/recognition/test'). Although no unknown label should be
            predicted, this evaluation is easier than the first one, since the
            classifier will contain less classes.
        '''

        # Check if at least one of the sets (training or unseen test) are being
        #   enrolled
        assert is_train_enrolled or is_test_enrolled, \
            '\'train/\' set or \'recognition/enroll/\' set must be enrolled!'
        
        # Open files containing the extracted embeddings from the three sets:
        #   training ('initial_enroll'), test to be enrolled ('new_enroll') and
        #   test to be evaluated ('new_enroll_test')
        with open(initial_enroll_path, 'rb') as f:
            initial_enroll = pickle.load(f)
        with open(new_enroll_path, 'rb') as f:
            new_enroll = pickle.load(f)
        with open(new_enroll_test_path, 'rb') as f:
            new_enroll_test = pickle.load(f)
            
        # First evaluation
        if is_train_enrolled and is_test_enrolled:

            # First, the embeddings, labels and classes refers to the ones at the
            #   training set
            self.embeddings = initial_enroll['embeddings']
            self.labels = initial_enroll['labels']
            self.classes = initial_enroll['classes']
            
            # The test set labels (which varies from 0 to 19) are shifted as much
            #   as the number of existing classes in the training set. As the
            #   training set contains 100 classes, the test set labels now will
            #   vary from 100 to 119
            new_enroll['labels'] = [l + len(self.classes)
                for l in new_enroll['labels']]
            new_enroll_test['labels'] = [l + len(self.classes)
                for l in new_enroll_test['labels']]
            
            # Then, the embeddings from the enroll test set are enrolled as well
            self.embeddings += new_enroll['embeddings']
            self.labels += new_enroll['labels']
            self.classes += new_enroll['classes']
        
        # Second evaluation
        elif is_train_enrolled:

            # The embeddings, labels and classes refers to the ones at the
            #   training set
            self.embeddings = initial_enroll['embeddings']
            self.labels = initial_enroll['labels']
            self.classes = initial_enroll['classes']
            
            # As the classifier will not contemplate samples refering to the test
            #   set, all samples' prediction must be refered to unknown. The
            #   unknown label is set to -1
            new_enroll_test['labels'] = [-1 for l in new_enroll_test['labels']]
            
        # Third evaluation
        elif is_test_enrolled:

            # The embeddings, labels and classes refers to the ones at the
            #   enroll test set. Also, no changes are made to the labels at the
            #   test set to be evaluated
            self.embeddings = new_enroll['embeddings']
            self.labels = new_enroll['labels']
            self.classes = new_enroll['classes']
        
        self.test_embeddings = new_enroll_test['embeddings']
        self.test_labels = new_enroll_test['labels']
            
        # Instantiate dog breed classifier from the acquired embeddings and
        #   labels
        self.create_classifier()
        
        # self.evaluate()
        
    def create_classifier(self):
        '''
        Creates a new dog breed classifier from the embeddings and labels to be
            enroller.
        '''

        # Instantiate a KNN classifier. Also considering the euclidean distance
        #   criteria for prediction
        self.classifier = KNeighborsClassifier(n_neighbors=5,
            weights='distance', metric='euclidean')

        # "Training" the classifier from the acquired embeddings and labels
        self.classifier.fit(self.embeddings, self.labels)
        
    def predict_label(self, dists, indexes, T):
        '''
        Predicts the dog breed of an input's embeddings regarding its nearest
            existing trained samples. Currently, based on "JÚNIOR, Pedro R. 
            Mendes et al. Nearest neighbors distance ratio open-set classifier.
            Machine Learning, v. 106, n. 3, p. 359-386, 2017.".
        '''

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
        '''
        Calculates the accuracy of the classifier on the testing set.
        '''

        # Get the distance ('test_dists') and indexes ('test_indexes') of the 'k'
        #   nearest neighbors of each instance of the test set
        test_dists, test_indexes = self.classifier.kneighbors(
            self.test_embeddings)
        
        # Get the dog breed label refering to each instance at 'test_indexes'
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

    evaluator_1 = EnrollerEvaluator(initial_enroll_path, new_enroll_path,
        new_enroll_test_path, is_train_enrolled=True, is_test_enrolled=True)
    accs_1 = evaluator_1.evaluate()
    
    evaluator_2 = EnrollerEvaluator(initial_enroll_path, new_enroll_path,
        new_enroll_test_path, is_train_enrolled=True, is_test_enrolled=False)
    accs_2 = evaluator_2.evaluate()
    
    evaluator_3 = EnrollerEvaluator(initial_enroll_path, new_enroll_path,
        new_enroll_test_path, is_train_enrolled=False, is_test_enrolled=True)
    accs_3 = evaluator_3.evaluate()
