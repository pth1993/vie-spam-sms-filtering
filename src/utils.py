from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn import linear_model
from sklearn.model_selection import KFold
import codecs
import numpy as np
from gensim import corpora, models


def load_corpus(filename):
    list_label = []
    list_content = []
    f = codecs.open(filename, 'r', 'utf-8')
    for line in f:
        line = line.strip().split('\t')
        list_label.append(line[0])
        list_content.append(line[1])
    return list_label, list_content


def doc_2_vec(content, label, vector, df_below=3, df_above=1.0, length=1):
    len_sms = []
    list_corpus = []
    list_corpus_new = []
    for line in content:
        line = line.split()
        list_corpus.append(line)
        len_sms.append(len(line))
    bigram = models.phrases.Phrases(list_corpus, min_count=10)
    list_corpus = list(bigram[list_corpus])
    for line in list_corpus:
        temp = []
        for item in line:
            item = item.split('_')
            if (len(item) > 1) and (len(set(item).intersection([u'date', u'phone', u'link', u'currency', u'emoticon'])) > 0):
                for word in item:
                    temp.append(word)
            elif len(set(item).intersection([u'date', u'phone', u'link', u'currency', u'emoticon'])) == 0:
                word = '_'.join(item)
                temp.append(word)
            else:
                temp.append(item[0])
        list_corpus_new.append(temp)
    dictionary = corpora.Dictionary(list_corpus_new)
    dictionary.filter_extremes(no_below=df_below, no_above=df_above, keep_n=100000)
    temp_corpus_bow = [dictionary.doc2bow(line) for line in list_corpus_new]
    content_bow = np.zeros((len(list_corpus), len(dictionary.keys())))
    for i in range(len(temp_corpus_bow)):
        for item in temp_corpus_bow[i]:
            content_bow[i][item[0]] = item[1]
    tfidf = models.TfidfModel(temp_corpus_bow)
    temp_corpus_tfidf = tfidf[temp_corpus_bow]
    content_tfidf = np.zeros((len(list_corpus), len(dictionary.keys())))
    for i in range(len(temp_corpus_tfidf)):
        for item in temp_corpus_tfidf[i]:
            content_tfidf[i][item[0]] = item[1]
    if vector == 'bow':
        content_vec = content_bow
    elif vector == 'tfidf':
        content_vec = content_tfidf
    len_sms = np.asarray(len_sms)
    len_sms = np.reshape(len_sms, (len(len_sms), 1))
    if length == 1:
        content_vec = np.concatenate((content_vec, len_sms), axis=1)
    le = preprocessing.LabelEncoder()
    label = le.fit_transform(label)
    return content_vec, label, len_sms, dictionary


def build_classifier_nb(content, label):
    clf = MultinomialNB().fit(content, label)
    return clf


def build_classifier_svm(content, label):
    clf = svm.LinearSVC(C=0.1).fit(content, label)
    return clf


def build_classifier_decisiontree(content, label):
    clf = tree.DecisionTreeClassifier().fit(content, label)
    return clf


def build_classifier_knn(content, label):
    clf = neighbors.KNeighborsClassifier().fit(content, label)
    return clf


def build_classifier_maxent(content, label):
    clf = linear_model.LogisticRegression().fit(content, label)
    return clf


def classifying_nb(content, content_vec, clf, label):
    label_predict = []
    probability_list = clf.predict_proba(content_vec)
    for i in range(len(probability_list)):
        if probability_list[i][1] > 0.9:
            label_predict.append(1)
        else:
            label_predict.append(0)
    for i in range(len(label)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    matrix = confusion_matrix(label, label_predict, labels=[1, 0])
    return matrix


def classifying_svm(content, content_vec, clf, label):
    label_predict = clf.predict(content_vec)
    for i in range(len(label)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    matrix = confusion_matrix(label, label_predict, labels=[1, 0])
    return matrix


def classifying_decisiontree(content, content_vec, clf, label):
    label_predict = clf.predict(content_vec)
    for i in range(len(label)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    matrix = confusion_matrix(label, label_predict, labels=[1, 0])
    return matrix


def classifying_knn(content, content_vec, clf, label):
    label_predict = clf.predict(content_vec)
    for i in range(len(label)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    matrix = confusion_matrix(label, label_predict, labels=[1, 0])
    return matrix


def classifying_maxent(content, content_vec, clf, label):
    label_predict = clf.predict(content_vec)
    for i in range(len(label)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    matrix = confusion_matrix(label, label_predict, labels=[1, 0])
    return matrix


def classifying_baseline(content, label):
    label_predict = np.zeros(len(label))
    for i in range(len(label)):
        if content[i][:2] in [u'qc', u'tb']:
            label_predict[i] = 1
    matrix = confusion_matrix(label, label_predict, labels=[1, 0])
    return matrix


# def k_fold_old(content, content_vec, label, classifier):
#     list_err_1 = []
#     list_err_2 = []
#     list_matrix = []
#     kf = KFold(len(label), n_folds=5, shuffle=True)
#     content = np.asarray(content)
#     list_index_train = []
#     list_index_test = []
#     for train_index, test_index in kf:
#         list_index_train.append(train_index)
#         list_index_test.append(test_index)
#         with open('list_index_train.pkl', 'wb') as output:
#             pickle.dump(list_index_train, output, pickle.HIGHEST_PROTOCOL)
#         with open('list_index_test.pkl', 'wb') as output:
#             pickle.dump(list_index_test, output, pickle.HIGHEST_PROTOCOL)
#         content_train, content_test = content[train_index], content[test_index]
#         content_vec_train, content_vec_test = content_vec[train_index], content_vec[test_index]
#         label_train, label_test = label[train_index], label[test_index]
#         if classifier == 'nb':
#             clf = build_classifier_nb(content_vec_train, label_train)
#             matrix = classifying_nb(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'svm':
#             clf = build_classifier_svm(content_vec_train, label_train)
#             matrix = classifying_svm(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'dt':
#             clf = build_classifier_decisiontree(content_vec_train, label_train)
#             matrix = classifying_decisiontree(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'knn':
#             clf = build_classifier_knn(content_vec_train, label_train)
#             matrix = classifying_knn(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'maxent':
#             clf = build_classifier_maxent(content_vec_train, label_train)
#             matrix = classifying_maxent(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'baseline':
#             matrix = classifying_baseline(content_test, label_test)
#         temp2 = matrix[0, 1]/float(sum(matrix[0]))
#         temp1 = matrix[1, 0]/float(sum(matrix[1]))
#         list_matrix.append(matrix)
#         list_err_1.append(temp1)
#         list_err_2.append(temp2)
#     return list_matrix, list_err_1, list_err_2
#
#
# def k_fold(content, content_vec, label, classifier):
#     list_false_positive = []
#     list_false_negative = []
#     list_true_positive = []
#     list_true_negative = []
#     list_matrix = []
#     content = np.asarray(content)
#     with open('list_index_train.pkl', 'rb') as input:
#         list_index_train = pickle.load(input)
#     with open('list_index_test.pkl', 'rb') as input:
#         list_index_test = pickle.load(input)
#     for i in range(len(list_index_train)):
#         train_index = list_index_train[i]
#         test_index = list_index_test[i]
#         content_train, content_test = content[train_index], content[test_index]
#         content_vec_train, content_vec_test = content_vec[train_index], content_vec[test_index]
#         label_train, label_test = label[train_index], label[test_index]
#         if classifier == 'nb':
#             clf = build_classifier_nb(content_vec_train, label_train)
#             matrix = classifying_nb(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'svm':
#             clf = build_classifier_svm(content_vec_train, label_train)
#             matrix = classifying_svm(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'dt':
#             clf = build_classifier_decisiontree(content_vec_train, label_train)
#             matrix = classifying_decisiontree(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'knn':
#             clf = build_classifier_knn(content_vec_train, label_train)
#             matrix = classifying_knn(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'maxent':
#             clf = build_classifier_maxent(content_vec_train, label_train)
#             matrix = classifying_maxent(content_test, content_vec_test, clf, label_test)
#         elif classifier == 'baseline':
#             matrix = classifying_baseline(content_test, label_test)
#         false_positive = matrix[1, 0]/float(sum(matrix[1]))
#         false_negative = matrix[0, 1]/float(sum(matrix[0]))
#         true_positive = matrix[0, 0]/float(sum(matrix[0]))
#         true_negative = matrix[1, 1]/float(sum(matrix[1]))
#         list_matrix.append(matrix)
#         list_false_positive.append(false_positive)
#         list_false_negative.append(false_negative)
#         list_true_positive.append(true_positive)
#         list_true_negative.append(true_negative)
#     return list_matrix, list_false_positive, list_false_negative, list_true_positive, list_true_negative


def evaluation(list_false_positive, list_false_negative, list_true_positive, list_true_negative):
    print 'False Positive Rate: ' + str(sum(list_false_positive)*20) + '%'
    print 'False Negative Rate: ' + str(sum(list_false_negative)*20) + '%'
    print 'True Positive Rate: ' + str(sum(list_true_positive)*20) + '%'
    print 'True Negative Rate: ' + str(sum(list_true_negative)*20) + '%'


def count_vocab(list_content):
    list_vocab = []
    list_corpus = []
    count = 1
    for line in list_content:
        line = line.split()
        list_vocab += line
        list_corpus.append(line)
        count += 1
    print len(list_vocab)
    temp = list(set(list_vocab))
    print len(temp)
    dictionary = corpora.Dictionary(list_corpus)
    print dictionary
    return temp, list_corpus


def classification(content_test, content_vec_train, content_vec_test, label_train, label_test, classifier):
    if classifier == 'nb':
        clf = build_classifier_nb(content_vec_train, label_train)
        matrix = classifying_nb(content_test, content_vec_test, clf, label_test)
    elif classifier == 'svm':
        clf = build_classifier_svm(content_vec_train, label_train)
        matrix = classifying_svm(content_test, content_vec_test, clf, label_test)
    elif classifier == 'dt':
        clf = build_classifier_decisiontree(content_vec_train, label_train)
        matrix = classifying_decisiontree(content_test, content_vec_test, clf, label_test)
    elif classifier == 'knn':
        clf = build_classifier_knn(content_vec_train, label_train)
        matrix = classifying_knn(content_test, content_vec_test, clf, label_test)
    elif classifier == 'maxent':
        clf = build_classifier_maxent(content_vec_train, label_train)
        matrix = classifying_maxent(content_test, content_vec_test, clf, label_test)
    elif classifier == 'baseline':
        matrix = classifying_baseline(content_test, label_test)
    false_positive = matrix[1, 0] / float(sum(matrix[1]))
    false_negative = matrix[0, 1] / float(sum(matrix[0]))
    true_positive = matrix[0, 0] / float(sum(matrix[0]))
    true_negative = matrix[1, 1] / float(sum(matrix[1]))
    return false_positive, false_negative, true_positive, true_negative


def kfold_classification(content, content_vec, label, classifier, fold):
    content = np.asarray(content)
    list_false_positive = []
    list_false_negative = []
    list_true_positive = []
    list_true_negative = []
    kf = KFold(n_splits=fold, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(label):
        content_train, content_test = content[train_index], content[test_index]
        content_vec_train, content_vec_test = content_vec[train_index], content_vec[test_index]
        label_train, label_test = label[train_index], label[test_index]
        false_positive, false_negative, true_positive, true_negative = \
            classification(content_test, content_vec_train, content_vec_test, label_train, label_test, classifier)
        list_false_positive.append(false_positive)
        list_false_negative.append(false_negative)
        list_true_positive.append(true_positive)
        list_true_negative.append(true_negative)
    return list_false_positive, list_false_negative, list_true_positive, list_true_negative
