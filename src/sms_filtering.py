from entity_tagging import entity_tagging
import utils
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="data directory")
parser.add_argument("--vectorize", help="vectorization method for texts: bow, tfidf")
parser.add_argument("--classifier", help="classification method: nb, svm, dt, knn, maxent, baseline")
parser.add_argument("--fold", help="fold number for cross-validation")
args = parser.parse_args()

start_time = datetime.now()
corpus_name = args.data_dir
vectorize = args.vectorize
classifier = args.classifier
fold = int(args.fold)
print "Loading data ..."
list_label, list_content = utils.load_corpus(corpus_name)
print "Tagging entity ..."
list_content = entity_tagging(list_content)
print "Converting document to vector ..."
list_content_vec, list_label, list_len_sms, dictionary = utils.doc_2_vec(list_content, list_label, vectorize)
print "Classifying..."
list_false_positive, list_false_negative, list_true_positive, list_true_negative = \
    utils.kfold_classification(list_content, list_content_vec, list_label, classifier, fold)
print "Evaluating..."
utils.evaluation(list_false_positive, list_false_negative, list_true_positive, list_true_negative)
end_time = datetime.now()
print "Running time: "
print (end_time - start_time)
