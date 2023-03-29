import argparse
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaTokenizer, RobertaForSequenceClassification
import transformers

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    accuracy = sk.metrics.accuracy_score(labels, 
                                         preds, 
                                         normalize=True, 
                                         sample_weight=None)    
    f1 = sk.metrics.f1_score(labels, 
                             preds, 
                             labels=None, 
                             pos_label=1, 
                             average='binary',
                             sample_weight=None)
    precision = sk.metrics.precision_score(labels, 
                                           preds, 
                                           labels=None, 
                                           pos_label=1, 
                                           average='binary',
                                           sample_weight=None)
    recall_scores = sk.metrics.recall_score(labels, 
                                            preds, 
                                            labels=None, 
                                            pos_label=1, 
                                            average='binary')
    return {"accuracy": accuracy, "f1 score": f1, "precision": precision, "recall scores":recall_scores}

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    tokenizer1 = RobertaTokenizer.from_pretrained("roberta-base")
    pretrained_roBerta_model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    
    return pretrained_roBerta_model

