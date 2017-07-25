# Vietnamese Spam SMS Filtering (vie-spam-sms-filtering)
-----------------------------------------------------------------
Code by **Thai-Hoang Pham** at Alt Inc. 

## 1. Introduction
**vie-spam-sms-filtering** is an implementation of the system described in a paper [Content-based Approach for 
Vietnamese Spam SMS Filtering](https://arxiv.org/abs/1705.04003). This system is used to filtering spam SMS in 
Vietnamese mobile operators and written by Python 2.7. In this system, we investigate several methods for detecting spam 
SMS in some aspects from vectorization to classification method. We use 5-fold cross-validation method for evaluation.

The performance of our system with pre-processing is described in the following table. 

|                | Without pre-processing    | With pre-processing |
|----------------|---------------------------|---------------------|
| True Positive  | 91.79%                    | 93.40%              |
| True Negative  | 99.69%                    | 99.60%              |
| False Positive | 0.31%                     | 0.40%               |
| False Negative | 8.21%                     | 6.60%               |

The performance of our system with each vector representation is described in the following table. 

|                | BOW    | TF-IDF |
|----------------|--------|--------|
| True Positive  | 93.40% | 84.66% |
| True Negative  | 99.60% | 99.58% |
| False Positive | 0.40%  | 0.42%  |
| False Negative | 6.60%  | 15.34% |

The performance of our system with each classifier is described in the following table. 

|                | SVM    | NB     | LR     | DT     | kNN    |
|----------------|--------|--------|--------|--------|--------|
| True Positive  | 93.40% | 95.15% | 92.99% | 92.18% | 78.13% |
| True Negative  | 99.60% | 96.25% | 99.64% | 99.12% | 99.64% |
| False Positive | 0.40%  | 3.75%  | 0.36%  | 0.88%  | 0.36%  |
| False Negative | 6.60%  | 9.85%  | 7.01%  | 7.82%  | 21.87% |

## 2. Installation

This software depends on NumPy, Scikit-learn, Gensim. You must have them installed before using **vie-ner-lstm**.

The simple way to install them is using pip:

```sh
	# pip install -U numpy scikit-learn gensim
```
## 3. Usage

### 3.1. Data

The input data's format is describe in the following table.

| Label | Message                                                                                                                                       |
|-------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Spam  | DatNen ThoCu 1OO% Chi 119tr/Nen, SoDo rieng, DoiDien KhuCongNghiep, BenhVien, TruongDaiHoc, KhuDanCu SamUat, gop 15th LS 0%. LH: 09497 59978. |
| Ham   | Tối nay 7h30 half life phố Phó Đức Chính nha anh em. Quán cũ :3                                                                               |                                                                        |

### 3.2. Command-line Usage

You can use vie-spam-sms-filtering software by a following command:

```sh
	$ bash sms_filtering.sh
```

Arguments in ``ner.sh`` script:

* ``data_dir``:       data directory
* ``vectorize``:         vector representation method (bow, tfidf)
* ``classifier``:   classification method (nb, svm, dt, knn, maxent, baseline)
* ``fold``:      fold number for cross-validation

## 4. References

[Thai-Hoang Pham, Phuong Le-Hong, "Content-based Approach for Vietnamese Spam SMS Filtering"](https://arxiv.org/abs/1705.04003)

## 5. Contact

**Thai-Hoang Pham** < phamthaihoang.hn@gmail.com >

Alt Inc, Hanoi, Vietnam
