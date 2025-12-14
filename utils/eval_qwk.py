import pandas as pd
import numpy as np
import re
import logging
from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_squared_error
from six import string_types
import argparse

class Evaluator():

    def __init__(self, traits):
        #self.target = target
        self.traits = traits

    def kappa(self, y_true, y_pred, weights=None, allow_off_by_one=False):
        
        logger = logging.getLogger(__name__)

        # Ensure that the lists are both the same length
        assert(len(y_true) == len(y_pred))

        # This rather crazy looking typecast is intended to work as follows:
        # If an input is an int, the operations will have no effect.
        # If it is a float, it will be rounded and then converted to an int
        # because the ml_metrics package requires ints.
        # If it is a str like "1", then it will be converted to a (rounded) int.
        # If it is a str that can't be typecast, then the user is
        # given a hopefully useful error message.
        # Note: numpy and python 3.3 use bankers' rounding.
        try:
            y_true = [int(np.round(float(y))) for y in y_true]
            y_pred = [int(np.round(float(y))) for y in y_pred]
        except ValueError as e:
            logger.error("For kappa, the labels should be integers or strings "
                        "that can be converted to ints (E.g., '4.0' or '3').")
            raise e

        # Figure out normalized expected values
        min_rating = min(min(y_true), min(y_pred))
        max_rating = max(max(y_true), max(y_pred))

        # shift the values so that the lowest value is 0
        # (to support scales that include negative values)
        y_true = [y - min_rating for y in y_true]
        y_pred = [y - min_rating for y in y_pred]

        # Build the observed/confusion matrix
        num_ratings = max_rating - min_rating + 1
        observed = confusion_matrix(y_true, y_pred,
                                    labels=list(range(num_ratings)))
        num_scored_items = float(len(y_true))

        # Build weight array if weren't passed one
        if isinstance(weights, string_types):
            wt_scheme = weights
            weights = None
        else:
            wt_scheme = ''
        if weights is None:
            weights = np.empty((num_ratings, num_ratings))
            for i in range(num_ratings):
                for j in range(num_ratings):
                    diff = abs(i - j)
                    if allow_off_by_one and diff:
                        diff -= 1
                    if wt_scheme == 'linear':
                        weights[i, j] = diff
                    elif wt_scheme == 'quadratic':
                        weights[i, j] = diff ** 2
                    elif not wt_scheme:  # unweighted
                        weights[i, j] = bool(diff)
                    else:
                        raise ValueError('Invalid weight scheme specified for '
                                        'kappa: {}'.format(wt_scheme))

        hist_true = np.bincount(y_true, minlength=num_ratings)
        hist_true = hist_true[: num_ratings] / num_scored_items
        hist_pred = np.bincount(y_pred, minlength=num_ratings)
        hist_pred = hist_pred[: num_ratings] / num_scored_items
        expected = np.outer(hist_true, hist_pred)

        # Normalize observed array
        observed = observed / num_scored_items

        # If all weights are zero, that means no disagreements matter.
        k = 1.0
        if np.count_nonzero(weights):
            k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

        return k
    
    def get_min_max_scores(self):
        return {
        1: {'overall': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word choice': (1, 6),
            'sentence fluency': (1, 6), 'conventions': (1, 6)},
        2: {'overall': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word choice': (1, 6),
            'sentence fluency': (1, 6), 'conventions': (1, 6)},
        3: {'overall': (0, 3), 'content': (0, 3), 'prompt adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'overall': (0, 3), 'content': (0, 3), 'prompt adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'overall': (0, 4), 'content': (0, 4), 'prompt adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'overall': (0, 4), 'content': (0, 4), 'prompt adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'overall': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6), 'style': (0,6)},
        8: {'overall': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word choice': (2, 12),
            'sentence fluency': (2, 12), 'conventions': (2, 12), 'voice': (2,12)}}


    def calc_kappa(self, pred, original, weight='quadratic'):
        kappa_score = self.kappa(original, pred, weights = weight)
        return kappa_score
    
    def calc_kappa_cohen(self, pred, original, weight='quadratic'):
        kappa_score = cohen_kappa_score(original, pred, weights = weight)
        return kappa_score
    
    def calc_rmse(self, pred, original):
        return mean_squared_error(original, pred)**0.5
    
    def mcrmse(self, pred, original):
        error = original - pred
        squared_error = np.square(error)
        colwise_mse = np.mean(squared_error, axis=0)
        root_colwise_mse = np.sqrt(colwise_mse)
        return root_colwise_mse
    

    def read_results(self, pred):
        results = pd.DataFrame(columns=self.traits)
        for i in range(len(pred)):
            text = str(pred[i])
            scores_list = text.split(", ") # , 기준으로 분리

            trait_scores = []
            for trait in self.traits:
                score = [re.search(r'\d+|nan', s) for s in scores_list if trait in s]
                if len(score)<1 or (None in score):
                    trait_score = -1
                else:
                    trait_score = score[0][0]
                    if len(score)>1:
                        print("%d idx text has %s score length %d."%(i,trait, len(score)))
                        print(score)
                trait_scores.append(trait_score)

            assert len(self.traits)==len(trait_scores), 'trait length error!'
            results.loc[i] = trait_scores
            
            results = results.astype('float')
            results = results.fillna(value=-1)
            
        return results
    
    def read_results_norm(self, pred):
        results = pd.DataFrame(columns=self.traits)
        for i in range(len(pred)):
            text = str(pred[i])
            scores_list = text.split(", ") # , 기준으로 분리

            trait_scores = []
            for trait in self.traits:
                score = [re.search(r'\d+.\d+|nan', s) for s in scores_list if trait in s]
                if len(score)<1 or (None in score):
                    trait_score = -1
                else:
                    trait_score = score[0][0]
                    if len(score)>1:
                        print("%d idx text has %s score length %d."%(i,trait, len(score)))
                        print(score)
                trait_scores.append(float(trait_score))

            assert len(self.traits)==len(trait_scores), 'trait length error!'
            results.loc[i] = trait_scores
            
            results = results.astype('float')
            results = results.fillna(value=-1)
            
        return results

    def evaluate(self, pred, target):
        results = self.read_results(pred)
        targets = self.read_results(target)
        
        qwk = {key: self.calc_kappa(results[key], targets[key]) for key in results.keys()}
        return qwk
    
    def evaluate_cohen(self, pred, target):
        results = self.read_results(pred)
        targets = self.read_results(target)
        
        qwk = {key: self.calc_kappa_cohen(results[key], targets[key]) for key in results.keys()}
        return qwk
    
    def evaluate_notnull_single(self, pred, target, trait):
        results = self.read_results(pred)
        targets = self.read_results(target)
        
        trait_tgt = targets[trait][targets[trait]!=-1]
        trait_pred = results[trait][trait_tgt.index] # evaluate only the samples whose ground-truth values are not null 
        
        qwk = self.calc_kappa(trait_pred, trait_tgt)
        print('Trait ',trait, 'QWK score: ', qwk)
        return qwk
    
    def evaluate_notnull(self, pred, target):
        results = self.read_results(pred)
        targets = self.read_results(target)
        
        qwk_results = {}
        for key in results.keys():
            trait_tgt = targets[key][targets[key]!=-1]
            print('len: ', len(trait_tgt))
            trait_pred = results[key][trait_tgt.index] # evaluate only the samples whose ground-truth values are not null
            qwk = self.calc_kappa(trait_pred, trait_tgt)
            qwk_results[key] = qwk

        return qwk_results
    
    def evaluate_notnull_norm(self, pred, target, prompt):
        results = self.read_results_norm(pred)
        targets = self.read_results_norm(target)
        
        min_max = self.get_min_max_scores()[prompt]
        
        qwk_results = {}
        for key in results.keys():
            trait_tgt = targets[key][targets[key]!=-1]
            print('len: ', len(trait_tgt))
            trait_pred = results[key][trait_tgt.index] # evaluate only the samples whose ground-truth values are not null
            
            min_val = min_max[key][0]
            max_val = min_max[key][1]
            
            trait_pred = trait_pred * (max_val-min_val) + min_val
            trait_tgt = trait_tgt * (max_val-min_val) + min_val
            
            qwk = self.calc_kappa(trait_pred, trait_tgt)
            qwk_results[key] = qwk

        return qwk_results
    
    def evaluate_notnull_bert(self, pred, target, prompt, trait):
        results = pred
        targets = target
        
        min_max = self.get_min_max_scores()[prompt]
        
        qwk_results = {}
        print('len: ', len(targets))
        
        min_val = min_max[trait][0]
        max_val = min_max[trait][1]
        
        trait_pred = results * (max_val-min_val) + min_val
        trait_tgt = targets * (max_val-min_val) + min_val
        
        qwk = self.calc_kappa(trait_pred, trait_tgt)
        qwk_results[trait] = qwk

        return qwk_results
    
    def evaluate_notnull_cohen(self, pred, target):
        results = self.read_results(pred)
        targets = self.read_results(target)
        
        qwk_results = {}
        for key in results.keys():
            trait_tgt = targets[key][targets[key]!=-1]
            print('len: ', len(trait_tgt))
            trait_pred = results[key][trait_tgt.index] # evaluate only the samples whose ground-truth values are not null
            qwk = self.calc_kappa_cohen(trait_pred, trait_tgt)
            qwk_results[key] = qwk

        return qwk_results
    
    def evaluate_mcrmse(self, pred, target):
        results = self.read_results(pred)
        targets = self.read_results(target)
        
        rmse_results = {}
        for key in results.keys():
            trait_tgt = targets[key][targets[key]!=-1]
            print('len: ', len(trait_tgt))
            trait_pred = results[key][trait_tgt.index] # evaluate only the samples whose ground-truth values are not null
            rmse = self.calc_rmse(trait_pred/2, trait_tgt/2) # 기존에 정수값 만들기 위해서 *2, thus 원래 값으로 예측하기 위해 /2
            rmse_results[key] = rmse
            
        mcrmse = np.mean([rmse_results[key] for key in rmse_results.keys()])
        
        return rmse, mcrmse
    