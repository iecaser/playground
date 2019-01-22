import collections
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

x = '我们是好朋友123'
r = jieba.cut(x)

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
c = CountVectorizer()
t = TfidfTransformer()
X = c.fit_transform(corpus)
XX = t.fit_transform(X)
s = SelectKBest(f_classif, 10)
Y = s.fit_transform(XX)
