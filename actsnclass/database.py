# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 10 August 2019
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import pandas as pd

from actsnclass.classifiers import random_forest
from actsnclass.query_strategies import uncertainty_sampling, random_sampling, boundary_sampling, min_spread
from actsnclass.metrics import get_snpcc_metric, get_confusion_matrix


__all__ = ['DataBase']


class DataBase:
    """DataBase object, upon which the active learning loop is performed.

    Attributes
    ----------
    classprob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    data: pd.DataFrame
        Complete information read from features files.
    features: pd.DataFrame
        Feature matrix to be used in classification (no metadata).
    features_names: list
        Header for attribute `features`.
    metadata: pd.DataFrame
        Features matrix which will not be used in classification.
    metadata_names: list
        Header for metadata.
    metrics_list_names: list
        Values for metric elements.
    predicted_class: np.array
        Predicted classes - results from ML classifier.
    queried_sample: list
        Complete information of queried objects.
    queryable_ids: np.array()
        Flag for objects available to be queried.
    test_features: pd.DataFrame
        Features matrix for the test sample.
    test_metadata: pd.DataFrame
        Metadata for the test sample
    test_labels: np.array()
        True classification for the test sample.
    train_features: pd.DataFrame
        Features matrix for the train sample.
    train_metadata: pd.DataFrame
        Metadata for the training sample.
    train_labels: np.array
        Classes for the training sample.

    Methods
    -------
    load_bazin_features(path_to_bazin_file: str)
        Load Bazin features from file
    load_features(path_to_file: str, method: str)
        Load features according to the chosen feature extraction method.
    build_samples(initial_training: str or int, nclass: int)
        Separate train and test samples.
    classify(method: str)
        Apply a machine learning classifier.
    evaluate_classification(metric_label: str)
        Evaluate results from classification.
    make_query(strategy: str, batch: int) -> list
        Identify new object to be added to the training sample.
    update_samples(query_indx: list)
        Add the queried obj(s) to training and remove them from test.
    save_metrics(loop: int, output_metrics_file: str)
        Save current metrics to file.
    save_queried_sample(queried_sample_file: str, loop: int, full_sample: str)
        Save queried sample to file.

    Examples
    --------
    >>> from actsnclass import DataBase

    Define the necessary paths

    >>> path_to_bazin_file = 'results/Bazin.dat'
    >>> metrics_file = 'results/metrics.dat'
    >>> query_file = 'results/query_file.dat'

    Initiate the DataBase object and load the data.
    >>> data = DataBase()
    >>> data.load_features(path_to_bazin_file, method='Bazin')

    Separate training and test samples and classify

    >>> data.build_samples(initial_training='original', nclass=2)
    >>> data.classify(method='RandomForest')
    >>> print(data.classprob)          # check predicted probabilities
    [[0.461 0.539]
    [0.346print(data.metrics_list_names)           # check metric header
    ['acc', 'eff', 'pur', 'fom']

    >>> print(data.metrics_list_values)          # check metric values
    [0.5975434599574068, 0.9024767801857585,
    0.34684684684684686, 0.13572404702012383] 0.654]
    ...
    [0.398 0.602]
    [0.396 0.604]]

    Calculate classification metrics

    >>> data.evaluate_classification(metric_label='snpcc')
    >>>

    Make query, choose object and update samples

    >>> indx = data.make_query(strategy='UncSampling', batch=1)
    >>> data.update_samples(indx)

    Save results to file

    >>> data.save_metrics(loop=0, output_metrics_file=metrics_file)
    >>> data.save_queried_sample(loop=0, queried_sample_file=query_file,
    >>>                          full_sample=False)
    """

    def __init__(self):
        self.classprob = np.array([])
        self.data = pd.DataFrame()
        self.features = pd.DataFrame([])
        self.features_names = []
        self.metadata = pd.DataFrame()
        self.metadata_names = []
        self.metrics_list_names = []
        self.metrics_list_values = []
        self.predicted_class = np.array([])
        self.queried_sample = []
        self.queryable_ids = np.array([])
        self.test_features = np.array([])
        self.test_metadata = pd.DataFrame()
        self.test_labels = np.array([])
        self.train_features = np.array([])
        self.train_metadata = pd.DataFrame()
        self.train_labels = np.array([])
        self.label_map = dict()
        self.cm = np.array([])
        self.feature_importances = np.array([])
        self.all_cms = list()

    def load_bazin_features(self, paths_to_bazin_files: list, screen=False):
        """Load Bazin features from file.

        Populate properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_bazin_file: str
            Complete path to Bazin features file.
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        """

        # read matrix with Bazin features
        self.data = pd.concat([pd.read_csv(j, sep=' ', index_col=False) for j in paths_to_bazin_files], ignore_index=True)
 
        # remove bad rows
        self.data = self.data.drop(columns='Unnamed: 40')
        self.data = self.data.replace(to_replace='None', value=np.nan).dropna()

        # list of features to use
        self.features_names = ['gA', 'gB', 'gt0', 'gtfall', 'gtrise', 'rA',
                               'rB', 'rt0', 'rtfall', 'rtrise', 'iA', 'iB',
                               'it0', 'itfall', 'itrise', 'zA', 'zB', 'zt0',
                               'ztfall', 'ztrise']

        self.metadata_names = ['id', 'redshift', 'type', 'sample']

        self.features = self.data[self.features_names]
        self.metadata = self.data[self.metadata_names]

        if screen:
            print('Loaded ', self.metadata.shape[0], ' samples!')
            
    def load_gp_features(self, paths_to_gp_files: list, screen=False):
        """Load Gaussian Process features from file.

        Populate properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_gp_file: str
            Complete path to GP features file.
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        """

        # read matrix with Bazin features
        self.data = pd.concat([pd.read_csv(j, sep=' ', index_col=False) for j in paths_to_gp_files], ignore_index=True)
        
        # remove bad rows
        self.data = self.data.drop(columns='Unnamed: 41')
        self.data = self.data.replace(to_replace='None', value=np.nan).dropna()

        # list of features to use
        if 'rbf1_l' in self.data.columns:
            self.features_names = ['host_photoz', 'host_photoz_err',\
                                    'pkmag_i', 'pos_flux_ratio', 'max_fr_blue', 'min_fr_blue',\
                                    'max_fr_red', 'min_fr_red', 'max_dt_yg', 'positive_width', 'time_fwd_max_20', 'time_fwd_max_50',\
                                    'time_bwd_max_20', 'time_bwd_max_50', 'time_fwd_max_20_ratio_blue',
                                    'time_fwd_max_50_ratio_blue', 'time_bwd_max_20_ratio_blue', 'time_bwd_max_50_ratio_blue',\
                                    'time_fwd_max_20_ratio_red', 'time_fwd_max_50_ratio_red', 'time_bwd_max_20_ratio_red',\
                                    'time_bwd_max_50_ratio_red', 'frac_s2n_5', 'frac_background', 'time_width_s2n_5',\
                                    'count_max_center', 'count_max_rise_20', 'count_max_rise_50', 'count_max_rise_100',\
                                    'count_max_fall_20', 'count_max_fall_50', 'count_max_fall_100', 'total_s2n',\
                                    'c1', 'rbf1_t', 'rbf1_l', 'c2', 'rbf2_t', 'rbf2_l', 'c3', 'rbf3_t',\
                                    'rbf3_l', 'loglike']
        
        elif 'matern_t' in self.data.columns:
            self.features_names = ['host_photoz', 'host_photoz_err',\
                                    'pkmag_i', 'pos_flux_ratio', 'max_fr_blue', 'min_fr_blue',\
                                    'max_fr_red', 'min_fr_red', 'max_dt_yg', 'positive_width', 'time_fwd_max_20', 'time_fwd_max_50',\
                                    'time_bwd_max_20', 'time_bwd_max_50', 'time_fwd_max_20_ratio_blue',
                                    'time_fwd_max_50_ratio_blue', 'time_bwd_max_20_ratio_blue', 'time_bwd_max_50_ratio_blue',\
                                    'time_fwd_max_20_ratio_red', 'time_fwd_max_50_ratio_red', 'time_bwd_max_20_ratio_red',\
                                    'time_bwd_max_50_ratio_red', 'frac_s2n_5', 'frac_background', 'time_width_s2n_5',\
                                    'count_max_center', 'count_max_rise_20', 'count_max_rise_50', 'count_max_rise_100',\
                                    'count_max_fall_20', 'count_max_fall_50', 'count_max_fall_100', 'total_s2n',\
                                    'c1', 'matern_t', 'matern_l', 'loglike']
            
        else:
            raise ValueError('Unexpected columns in data.')
            
        self.metadata_names = ['id', 'redshift', 'type', 'sample']

        self.features = self.data[self.features_names]
        self.metadata = self.data[self.metadata_names]

        if screen:
            print('Loaded ', self.metadata.shape[0], ' samples!')            

    def load_features(self, paths_to_files: list, method='Bazin',
                      screen=True):
        """Load features according to the chosen feature extraction method.

        Populates properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_file: str
            Complete path to features file.
        method: str (optional)
            Feature extraction method. The current implementation only
            accepts method=='Bazin'
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        """

        if method == 'Bazin':
            self.load_bazin_features(paths_to_files, screen=screen)
        elif method == 'GP':
            self.load_gp_features(paths_to_files, screen=screen)
        else:
            raise ValueError('Only Bazin and GP features are implemented! '
                             '\n Feel free to add other options.')

    def build_samples(self, initial_training='original', nclass=2,
                      screen=True):
        """Separate train and test samples.

        Populate properties: train_features, train_header, test_features,
        test_header, queryable_ids (if flag available), train_labels and
        test_labels.

        Parameters
        ----------
        initial_training : str or int
            Choice of initial training sample.
            If 'original': begin from the train sample flagged in the file
            If int: choose the required number of samples at random,
            ensuring that at least half are SN Ia.
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        screen: bool (optional)
            If True display the dimensions of training and test samples.
        """

        # separate original training and test samples
        if initial_training == 'original' and nclass == 2:
            train_flag = self.metadata['sample'] == 'train'
            train_data = self.features[train_flag]
            self.train_features = train_data.values
            self.train_metadata = self.metadata[train_flag]

            test_flag = np.logical_or(self.metadata['sample'] == 'test',
                                      self.metadata['sample'] == 'queryable')
            test_data = self.features[test_flag]
            self.test_features = test_data.values
            self.test_metadata = self.metadata[test_flag]

            if 'queryable' in self.metadata['sample'].values:
                queryable_flag = self.metadata['sample'].values == 'queryable'
                self.queryable_ids = self.metadata[queryable_flag]['id'].values
            else:
                self.queryable_ids = self.test_metadata['id'].values

            if nclass == 2:
                train_ia_flag = self.train_metadata['type'] == 'Ia'
                self.train_labels = np.array([int(item) for item in train_ia_flag])

                test_ia_flag = self.test_metadata['type'] == 'Ia'
                self.test_labels = np.array([int(item) for item in test_ia_flag])         
        
        # separate training and test samples from scratch
        elif isinstance(initial_training, int) and nclass == 2:

            # initialize the temporary label holder
            train_indexes = np.array([1])
            temp_labels = np.array([self.metadata['type'].values[train_indexes]])

            # make sure half are Ias and half are non-Ias
            while sum(temp_labels == 'Ia') != initial_training // 2 + 1:
                # this is an array of 5 indexes
                train_indexes = np.random.randint(low=0,
                                                  high=self.metadata.shape[0],
                                                  size=initial_training)
                temp_labels = self.metadata['type'].values[train_indexes]

            # set training
            train_flag = self.metadata['type'].values[train_indexes] == 'Ia'
            self.train_labels = np.array([int(item) for item in train_flag])
            self.train_features = self.features.values[train_indexes]
            self.train_metadata = pd.DataFrame(self.metadata.values[train_indexes],
                                               columns=self.metadata_names)

            # set test set as all objs apart from those in training
            test_indexes = np.array([i for i in range(self.metadata.shape[0])
                                     if i not in train_indexes])
            test_ia_flag = self.metadata['type'].values[test_indexes] == 'Ia'
            self.test_labels = np.array([int(item) for item in test_ia_flag])
            self.test_features = self.features.values[test_indexes]
            self.test_metadata = pd.DataFrame(self.metadata.values[test_indexes],
                                              columns=self.metadata_names)

            if 'queryable' in self.metadata['sample'].values:
                test_flag = np.array([True if i in test_indexes else False
                                      for i in range(self.metadata.shape[0])])
                queryable_flag = self.metadata['sample'] == 'queryable'
                combined_flag = np.logical_and(test_flag, queryable_flag)
                self.queryable_ids = self.metadata[combined_flag]['id'].values
            else:
                self.queryable_ids = self.test_metadata['id'].values                

        elif initial_training == 'original' and nclass > 2:

            train_indexes = self.metadata['sample'] == 'train'

            for k, sntype in enumerate(np.unique(self.metadata['type'])):
                self.label_map[sntype] = k
                
            assert len(np.unique(self.metadata['type'])) == nclass,\
            'Mismatch between number of classes specified and number of unique types in your data.'
            
            # set training
            train_flag = [self.label_map[k] for k in self.metadata['type'].values[train_indexes]]
            self.train_labels = np.array([int(item) for item in train_flag])
            self.train_features = self.features.values[train_indexes]
            self.train_metadata = pd.DataFrame(self.metadata.values[train_indexes],
                                               columns=self.metadata_names)

            # set test set as all objs apart from those in training
            test_indexes = np.logical_or(self.metadata['sample'] == 'test',
                                      self.metadata['sample'] == 'queryable')
            test_ia_flag = [self.label_map[k] for k in self.metadata['type'].values[test_indexes]]
            self.test_labels = np.array([int(item) for item in test_ia_flag])
            self.test_features = self.features.values[test_indexes]
            self.test_metadata = pd.DataFrame(self.metadata.values[test_indexes],
                                              columns=self.metadata_names)

            if 'queryable' in self.metadata['sample'].values:
                test_flag = np.array([True if i in test_indexes else False
                                      for i in range(self.metadata.shape[0])])
                queryable_flag = self.metadata['sample'] == 'queryable'
                combined_flag = np.logical_and(test_flag, queryable_flag)
                self.queryable_ids = self.metadata[combined_flag]['id'].values
            else:
                self.queryable_ids = self.test_metadata['id'].values
                
        elif isinstance(initial_training, int) and nclass > 2:

            train_indexes = np.random.randint(low=0,
                                              high=self.metadata.shape[0],
                                              size=initial_training)

            for k, sntype in enumerate(np.unique(self.metadata['type'])):
                self.label_map[sntype] = k
                
            assert len(np.unique(self.metadata['type'])) == nclass,\
            'Mismatch between number of classes specified and number of unique types in your data.'
            
            # set training
            train_flag = [self.label_map[k] for k in self.metadata['type'].values[train_indexes]]
            self.train_labels = np.array([int(item) for item in train_flag])
            self.train_features = self.features.values[train_indexes]
            self.train_metadata = pd.DataFrame(self.metadata.values[train_indexes],
                                               columns=self.metadata_names)

            # set test set as all objs apart from those in training
            test_indexes = np.array([i for i in range(self.metadata.shape[0])
                                     if i not in train_indexes])
            test_ia_flag = [self.label_map[k] for k in self.metadata['type'].values[test_indexes]]
            self.test_labels = np.array([int(item) for item in test_ia_flag])
            self.test_features = self.features.values[test_indexes]
            self.test_metadata = pd.DataFrame(self.metadata.values[test_indexes],
                                              columns=self.metadata_names)

            if 'queryable' in self.metadata['sample'].values:
                test_flag = np.array([True if i in test_indexes else False
                                      for i in range(self.metadata.shape[0])])
                queryable_flag = self.metadata['sample'] == 'queryable'
                combined_flag = np.logical_and(test_flag, queryable_flag)
                self.queryable_ids = self.metadata[combined_flag]['id'].values
            else:
                self.queryable_ids = self.test_metadata['id'].values
                
        else:
            raise ValueError('"Initial training" should be '
                             '"original" or integer.')

        if screen:
            print('Training set size: ', self.train_metadata.shape[0])
            print('Test set size: ', self.test_metadata.shape[0])

    def classify(self, method='RandomForest', n_jobs = 1, feature_importances = False):
        """Apply a machine learning classifier.

        Populate properties: predicted_class and class_prob

        Parameters
        ----------
        method: str (optional)
            Chosen classifier.
            The current implementation only accepts `RandomForest'.
        """
        
        if method == 'RandomForest':
            if feature_importances:
                self.predicted_class,  self.classprob, self.feature_importances = \
                    random_forest(self.train_features, self.train_labels,
                                  self.test_features, n_jobs = n_jobs,
                                  feature_importances = feature_importances)
            else:
                self.predicted_class,  self.classprob = \
                    random_forest(self.train_features, self.train_labels,
                                  self.test_features, n_jobs = n_jobs)          

        else:
            raise ValueError('Only RandomForest classifier is implemented!'
                             '\n Feel free to add other options.')

    def evaluate_classification(self, metric_label='snpcc'):
        """Evaluate results from classification.

        Populate properties: metric_list_names and metrics_list_values.

        Parameters
        ----------
        metric_label: str
            Choice of metric. Currenlty only `snpcc` is accepted.
        """

        if metric_label == 'snpcc':
            self.metrics_list_names, self.metrics_list_values = \
                get_snpcc_metric(list(self.predicted_class),
                                 list(self.test_labels))
        elif metric_label == 'cm':
            self.cm = get_confusion_matrix(y_true = self.test_labels,\
                                          y_pred = self.predicted_class)
            self.all_cms.extend(get_confusion_matrix(y_true = self.test_labels,\
                                          y_pred = self.predicted_class))
        else:
            raise ValueError('Only snpcc and cm metrics are implemented!'
                             '\n Feel free to add other options.')

    def make_query(self, strategy='UncSampling', batch=1, nclass=8,
                   dump=False) -> list:
        """Identify new object to be added to the training sample.

        Parameters
        ----------
        strategy: str (optional)
            Strategy used to choose the most informative object.
            Current implementation accepts 'UncSampling' and
            'RandomSampling'. Default is `UncSampling`.

        batch: int (optional)
            Number of objects to be chosen in each batch query.
            Default is 1.
        dump: bool (optional)
            If true, display on screen information about the
            displacement in order and classificaion probability due to
            constraints on queryable sample.

        Returns
        -------
        query_indx: list
            List of indexes identifying the objects to be queried in decreasing
            order of importance.
            If strategy=='RandomSampling' the order is irrelevant.
        """

        if strategy == 'UncSampling':
            query_indx = uncertainty_sampling(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.test_metadata['id'].values,
                                              batch=batch, dump=dump, nclass=nclass)
            return query_indx
        
        elif strategy == 'boundary_Ia':
            
            boundary_idx = self.label_map['Ia']
            
            query_indx = boundary_sampling(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.test_metadata['id'].values,
                                              label_map=self.label_map, boundary_class=boundary_idx,
                                              batch=batch, dump=dump, nclass=nclass)
            return query_indx
        
        elif strategy == 'min_spread':
            
            query_indx = min_spread(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.test_metadata['id'].values,
                                              batch=batch, dump=dump, nclass=nclass)
            
            return query_indx
        

        elif strategy == 'RandomSampling':
            query_indx = random_sampling(queryable_ids=self.queryable_ids,
                                         test_ids=self.test_metadata['id'].values,
                                         batch=batch,nclass=nclass)

            for n in query_indx:
                if self.test_metadata['id'].values[n] not in self.queryable_ids:
                    raise ValueError('Chosen object is not available for query!')

            return query_indx

        else:
            raise ValueError('Invalid strategy. Only "UncSampling" and '
                             '"RandomSampling are implemented! \n '
                             'Feel free to add other options. ')

    def update_samples(self, query_indx: list, loop: int, epoch=0):
        """Add the queried obj(s) to training and remove them from test.

        Update properties: train_headers, train_features, train_labels,
        test_labels, test_headers and test_features.

        Parameters
        ----------
        query_indx: list
            List of indexes identifying objects to be moved.
        loop: int
            Store number of loop when this query was made.
        """

        all_queries = []
        for obj in query_indx:
            # add object to the query sample
            query_header = self.test_metadata.values[obj]
            query_features = self.test_features[obj]
            line = [epoch]
            for item in query_header:
                line.append(item)
            for item1 in query_features:
                line.append(item1)

            # add object to the training sample
            new_header = pd.DataFrame([query_header], columns=self.metadata_names)
            self.train_metadata = pd.concat([self.train_metadata, new_header], axis=0,
                                            ignore_index=True)
            self.train_features = np.append(self.train_features,
                                            np.array([self.test_features[obj]]),
                                            axis=0)
            self.train_labels = np.append(self.train_labels,
                                          np.array([self.test_labels[obj]]),
                                          axis=0)

            # remove queried object from test sample
            self.test_metadata = self.test_metadata.drop(self.test_metadata.index[obj])
            self.test_labels = np.delete(self.test_labels, obj, axis=0)
            self.test_features = np.delete(self.test_features, obj, axis=0)

            all_queries.append(line)

        # update queried samples
        self.queried_sample.append(all_queries)

    def save_metrics(self, loop: int, output_metrics_file: str, epoch: int, batch=1, metric_label='snpcc'):
        """Save current metrics to file.

        If loop == 0 the 'output_metrics_file' will be created or overwritten.
        Otherwise results will be added to an existing 'output_metrics file'.

        Parameters
        ----------
        loop: int
            Number of learning loops finished at this stage.
        output_metrics_file: str
            Full path to file to store metrics results.
        batch: int
            Number of queries in each loop.
        epoch: int
            Days since the beginning of the survey.
        """

        if metric_label == 'snpcc':
            # add header to metrics file
            if not os.path.exists(output_metrics_file) or loop == 0:
                with open(output_metrics_file, 'w') as metrics:
                    metrics.write('loop ')
                    for name in self.metrics_list_names:
                        metrics.write(name + ' ')
                    for j in range(batch):
                        metrics.write('query_id' + str(j + 1))
                    metrics.write('\n')

            # write to file
            with open(output_metrics_file, 'a') as metrics:
                metrics.write(str(epoch) + ' ')
                for value in self.metrics_list_values:
                    metrics.write(str(value) + ' ')
                for j in range(batch):
                    metrics.write(str(self.queried_sample[loop][j][1]) + ' ')
                metrics.write('\n')
                
        elif metric_label == 'cm':
            np.savetxt(output_metrics_file,self.all_cms)
            
        else:
            raise ValueError('Invalid metric label.')

    def save_queried_sample(self, queried_sample_file: str, loop: int,
                            full_sample=False, batch=1):
        """Save queried sample to file.

        Parameters
        ----------
        queried_sample_file: str
            Complete path to output file.
        loop: int
            Number of learning loops finished at this stage.
        full_sample: bool (optional)
            If true, write down a complete queried sample stored in
            property 'queried_sample'. Otherwise append 1 line per loop to
            'queried_sample_file'. Default is False.
        """

        if full_sample:
            full_header = self.metadata_names + self.features_names
            query_sample = pd.DataFrame(self.queried_sample, columns=full_header)
            query_sample.to_csv(queried_sample_file, sep=' ', index=False)

        elif isinstance(loop, int):
            if not os.path.exists(queried_sample_file) or loop == 0:
                # add header to query sample file
                full_header = self.metadata_names + self.features_names
                with open(queried_sample_file, 'w') as query:
                    query.write('day ')
                    for item in full_header:
                        query.write(item + ' ')
                    query.write('\n')

            # save query sample to file
            with open(queried_sample_file, 'a') as query:
                for batch in range(batch):
                    for elem in self.queried_sample[loop][batch]:
                        query.write(str(elem) + ' ')
                    query.write('\n')


def main():
    return None


if __name__ == '__main__':
    main()
