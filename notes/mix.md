# perf 
[Perf -- Linux下的系统性能调优工具 part1](https://www.ibm.com/developerworks/cn/linux/l-cn-perf1/index.html)

# typing
```
from typing import Tuple, List, Dict, Any, NamedTuple
E = NamedTuple(
    'E', [('input_name', str),
          ('key', str)]
)
```
https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple
[typing类型提示](https://yiyibooks.cn/xx/python_352/library/typing.html)

# SQL
## Hive import csv file
```
hive -e "CREATE TABLE tmp.vender_level
(
  vender_id STRING,
  level STRING
)
  ROW FORMAT DELIMITED
  FIELDS TERMINATED BY ','
  LINES TERMINATED BY '\n';
  LOAD DATA LOCAL INPATH '/home/mart_vda/zxf/jdd/hive/vender_level_20181210.csv' OVERWRITE INTO TABLE tmp.vender_level;"

  # TEST pass
  # hive -e "select * from tmp.vender_level" > test.csv
```
## group by
```
...
group by
  user_log_acct,
  batch_id,
  cps_valid_begin_tm,
  parent_sale_ord_id,
  cps_valid_end_tm
  ...
```
`group by` 后面为一个整体, 全部一致才是一组, 顺序无所谓;

# python

## functools.partial(func[,*args][, **keywords])
> Roughly equivalent to:

```
def partial(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*(args + fargs), **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc
```


## warning
This is not recommend because it interrupt!
```
if xxx:
  raise Warning('some warning...')
```
- [stackoverflow:Raise warning in Python without interrupting program](https://stackoverflow.com/questions/3891804/raise-warning-in-python-without-interrupting-program)

## copy
```
import copy
copy.copy()
copy.deepcopy()

# lgbm input params dict will be changed to default!!
# so do the follows:
self.params = copy.deepcopy(params)
```
[Python拷贝(深拷贝deepcopy与浅拷贝copy)](https://www.cnblogs.com/Richardzhu/p/4723750.html)

## syntax

### decorator

```python
# source code from keras_applications
def keras_modules_injection(base_fun):
  """Decorator injecting tf.keras replacements for Keras modules.

  Arguments:
      base_fun: Application function to decorate (e.g. `MobileNet`).

  Returns:
      Decorated function that injects keyword argument for the tf.keras
      modules required by the Applications.
  """

  def wrapper(*args, **kwargs):
    if hasattr(keras_applications, 'get_submodules_from_kwargs'):
      kwargs['backend'] = backend
      if 'layers' not in kwargs:
        kwargs['layers'] = layers
      kwargs['models'] = models
      kwargs['utils'] = utils
    return base_fun(*args, **kwargs)
  return wrapper

# vgg16.VGG16 is class; 这样实现了通用参数设定(包装)
@keras_modules_injection
def VGG16(*args, **kwargs):
  return vgg16.VGG16(*args, **kwargs)
```
[python decorator](https://realpython.com/primer-on-python-decorators/)

###
```
# Do not use ugly """some string""" because indent!
f.write('line1\n'
        'line2\n'
        'line3\n'
        )
a = ('God'
     'help'
     'me')
b = 'hello'\
    'world'\
    ':)'
```

### dict

```python
# call as funtion
dict(a=1,b=2,c=3)
Out[71]: {'a': 1, 'b': 2, 'c': 3}

# source code from tf offical example
print("Training set accuracy: {accuracy}".format(**train_eval_result))**))
```
### setdefault
Python 字典 setdefault() 函数和get() 方法类似, 如果键不存在于字典中，将会添加键并将值设为默认值。
- 参考
[参考](http://www.runoob.com/python/att-dictionary-setdefault.html)

### NEVER USE STATICMETHOD!

> Never use @staticmethod unless forced to in order to integrate with an API defined in an existing library. Write a module level function instead.
> Use @classmethod only when writing a named constructor or a class-specific routine that modifies necessary global state such as a process-wide cache.
- [Google Python Style](http://google.github.io/styleguide/pyguide.html)

> We all know how limited static methods are. (They’re basically an accident — back in the Python 2.2
>  days when I was inventing new-style classes and descriptors, I meant to implement class methods but
>  at first I didn’t understand them and accidentally implemented static methods first. Then it was too
>  late to remove them and only provide class methods.
- [When to use Static Methods in Python? Never](https://www.webucator.com/blog/2016/05/when-to-use-static-methods-in-python-never/)
- [Guido van Rossum](https://mail.python.org/pipermail/python-ideas/2012-May/014969.html)


### lambda
Example from open-solution-ShipDetection:
1. `self._initialize_model_weights = lambda: None`
2. `parameter_list = [filter(lambda p: p.requires_grad, model.parameters())]`

### super()
[知乎super](https://www.zhihu.com/question/20040039)

### self.__class__.__name__
class name

### 大坑！！！
python中`a += b` IS NOT `a = a + b`!!!
当a为list或者np等对象时，多次调用改变a将造成大坑！
[知乎问答python中a+=b和a=a+b区别](https://www.zhihu.com/question/20114936)

### __dict__
```python
class A:
    x = 0
    def __init__(self):
        self.a = 1
        self.b = 2
a = A()
a.__dict__
# only a,b, no x
```

### func is object too
```
def f():
    pass
f.x = 1
```

### dict.pop()
```
d = {'a':1,'b':2}
d.pop('a')
d ==> {'b':2}
```

### isinstance
```
# example 1(lightgbm):
def is_numpy_1d_array(data):
    """Check is 1d numpy array"""
    return isinstance(data, np.ndarray) and len(data.shape) == 1

# example 2(pytorch official tutorial):
assert isinstance(output_size, (int, tuple))
```

### raise
```
raise TypeError('Need Model file or Booster handle to create a predictor')
```

### @stacticmethod @clasmethod

http://funhacks.net/explore-python/Class/method.html

## operator.attrgetter
```
callbacks_before_iter = {f,f1,..}
# f has attr `order`
callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
```
[operator — Standard operators as functions¶](https://docs.python.org/2/library/operator.html)
[Python中的sorted函数以及operator.itemgetter函数](https://blog.csdn.net/dongtingzhizi/article/details/12068205)

## sys
### python version
```
In[9]: sys.version
Out[9]: '3.5.4 (default, Sep 10 2018, 11:57:25) \n[GCC 5.3.0]'
In[10]: sys.version_info
Out[10]: sys.version_info(major=3, minor=5, micro=4, releaselevel='final', serial=0)
In[11]: sys.version_info[0]
Out[11]: 3
```

## os
### makedirs vs. mkdir
[stackoverflow](https://stackoverflow.com/questions/13819496/what-is-different-between-makedirs-and-mkdir-of-os)

### os.path.dirname
```
import os
d = os.path.dirname(__file__)
print(d)
```

# steppy
adapter E deeper..深层映射

# pandas

## to_csv()
```
# 如此指定编码可加入BOM头信息, excel打开不乱码!! 否则正常文本形式能打开, 但是excel打开就会乱码
msg_100k.to_csv(msg_clean_filepath, index=False, encoding='utf_8_sig')
```

## Basic Reminder

- `pd.Series.unique()`
- `pd.Series.count()`

## pd.read_csv()

`r = pd.read_csv('data_nanjing_2018.12.13/input/nj_order_180.csv', error_bad_lines=False)`
> error_bad_lines : boolean, default True
    Lines with too many fields (e.g. a csv line with too many commas) will by
    default cause an exception to be raised, and no DataFrame will be returned.
    If False, then these "bad lines" will dropped from the DataFrame that is
    returned.

## pd.Series.map
```
def map(self, arg, na_action=None):
    """
    Map values of Series using input correspondence (a dict, Series, or
    function).

    Parameters
    ----------
    arg : function, dict, or Series
        Mapping correspondence.
    na_action : {None, 'ignore'}
        If 'ignore', propagate NA values, without passing them to the
        mapping correspondence.

    Returns
    -------
    y : Series
        Same index as caller.

# example:
def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df
```
- [kaggle elo-world kernel](https://www.kaggle.com/fabiendaniel/elo-world)

## query
> expr : string
>
>    The query string to evaluate. You can refer to variables in the environment by prefixing them with an ‘@’ character like @a + b.

```
filename = '...'
number_of_ships = get_number_of_ships(annotations.query('ImageId == @filename'))
```
[pandas.DataFrame.query](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html)

## apply
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html
## map
[series.map](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html)
## 内存优化
- https://zhuanlan.zhihu.com/p/28531346
- 注意从`read_csv`的时候就可以指定数据类型！
## 速度优化
- https://python.freelycode.com/contribution/detail/1083


## replace
```
df.replace([np.inf, -np.inf], np.nan)
```
- [stackoverflow: dropping infinite values from dataframes in pandas?](https://stackoverflow.com/questions/17477979/dropping-infinite-values-from-dataframes-in-pandas)

## dtypes
```python
df.dtypes.kind
df.select_types(include=['xxxx'])

# change some columns' dtypes
df[['parks', 'playgrounds', 'sports']].apply(lambda x: x.astype('category'))
df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
```
[Python Pandas - Changing some column types to categories](https://stackoverflow.com/questions/28910851/python-pandas-changing-some-column-types-to-categories)
[check dtype of df](https://stackoverflow.com/questions/22697773/how-to-check-the-dtype-of-a-column-in-python-pandas)

## pd.cut
```
pd.cut(df['col'],[0,500,1000],labels=['0~500','500~1000'])
# 注意切分出的区间类型为IntervalIndex
```
[pd离散化和面元划分](https://zhuanlan.zhihu.com/p/33441181)

## pd.IntervalIndex
```
pd.IntervalIndex.from_breaks([0,1,2,30])
```


## rename
```
df = df.rename(columns=str)
```

## df.select_dtypes
```
# select_dtypes按照dtypes filter
cat_cols = data.select_dtypes(include=['category']).columns
```

## loc/iloc/ix
1. 注意区分`iloc`和`loc`
2. 为了避免出错，尽量不用`ix`
3. iloc[]
```
n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
```
iloc[rows,cols]切片！！
[Pandas——ix vs loc vs iloc区别](https://blog.csdn.net/Xw_Classmate/article/details/51333646)

## DataFrame.reindex

```
DataFrame.reindex(labels=None, index=None, columns=None, axis=None, method=None, copy=True, level=None, fill_value=nan, limit=None, tolerance=None)[source]
# reindex columns example:
df.reindex(['a','b','c'], axis=1)
```
[DataFrame.reindex](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reindex.html)

## reset_index
```
df.reset_index(level=['a','b'])
```
[stack overflow](https://stackoverflow.com/questions/20461165/how-to-convert-pandas-index-in-a-dataframe-to-a-column)

## join / set_index
ataFrame.join always uses other’s index but we can use any column in the caller. This method preserves the original caller’s index in the result.
```
caller.join(other.set_index('key'), on='key')
```
[pandas.DataFrame.join](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html)

## DataFrame.sample
```
DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
```
[DataFrame.sample](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html)


# python: numbers

About `numbers.Integral`
'numbers.Integral' & 'np.integer'
[stack overflow: numbers](https://stackoverflow.com/questions/8203336/difference-between-int-and-numbers-integral-in-python)

# numpy
## np.stack

```
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.stack((a, b))
    array([[1, 2, 3],
           [2, 3, 4]])
>>> np.stack((a, b), axis=-1)
    array([[1, 2],
           [2, 3],
           [3, 4]])
```

[np.stack](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.stack.html)

## np.ptp
```
def ptp(a, axis=None, out=None):
    """
    Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for 'peak to peak'.

    Parameters
    ----------
    a : array_like
        Input values.
    axis : int, optional
        Axis along which to find the peaks.  By default, flatten the
        array.
    out : array_like
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type of the output values will be cast if necessary.

    Returns
    -------
    ptp : ndarray
        A new array holding the result, unless `out` was
        specified, in which case a reference to `out` is returned.
```
## np.percentile

```
def percentile(a, q, axis=None, out=None,
               overwrite_input=False, interpolation='linear', keepdims=False):
    """
    Compute the qth percentile of the data along the specified axis.

    Returns the qth percentile(s) of the array elements.
```


## np.iinfo/np.finfo
```
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```
- [kaggle elo-world kernel](https://www.kaggle.com/fabiendaniel/elo-world)

## np.random
### np.random.shuffle(x)
NOTE shuffle returns `None`! It is in-place operation.
```
Parameters
----------
x : array_like
    The array or list to be shuffled.

Returns
-------
None
    
```

### np.random.choice(a, size=None, replace=True, p=None)
```
>>> np.random.choice(5, 3)
array([0, 3, 4])
>>> #This is equivalent to np.random.randint(0,5,3)
# NOTE size can greater than a.shape
```

## np.where
`where(condition, x=None, y=None)`
``np.where(x>0)`` return value with the same shape of ``x``

## np.random.RandomState(1234) & np.random.seed(1234)
### seed

```
rng = np.random.RandomState(1234)
rng.permutation(...)
rng.randn(...)
rng.shuffle(...) # shuffle is in-place operation
# or
np.random.seed(1234)
np.random.permutation(...)
np.random.randn(...)

# source code from sklearn
def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
```
[rng.shuffle](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.shuffle.html)

### permutation

when `rng` is specified, rng.permutation产生相同的排列序列！！（注意每次不同，但出现顺序是一致的）
```
In [160]: a=np.random.RandomState(123)

In [161]: for i in range(6):print(a.permutation(10))
[4 0 7 5 8 3 1 6 9 2]
[3 5 4 2 8 7 6 9 0 1]
[2 1 8 7 0 4 5 6 3 9]
[5 1 0 8 6 9 4 2 3 7]
[6 8 2 9 5 4 3 1 7 0]
[9 4 7 0 1 2 3 8 6 5]

In [162]: b=np.random.RandomState(123)

In [163]: for i in range(6):print(b.permutation(10))
[4 0 7 5 8 3 1 6 9 2]
[3 5 4 2 8 7 6 9 0 1]
[2 1 8 7 0 4 5 6 3 9]
[5 1 0 8 6 9 4 2 3 7]
[6 8 2 9 5 4 3 1 7 0]
[9 4 7 0 1 2 3 8 6 5]
```
## np.unique

```
In [171]: np.unique(['a','b','c','a','a','b','b'], return_inverse=True)
Out[171]: (array(['a', 'b', 'c'], dtype='<U1'), array([0, 1, 2, 0, 0, 1, 1]))
```

## numpy.bincount(x, weights=None, minlength=0)

```
>>> np.bincount(np.arange(5))
array([1, 1, 1, 1, 1])
>>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
array([1, 3, 1, 1, 0, 0, 0, 1])
>>>
>>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
>>> np.bincount(x).size == np.amax(x)+1
True

# source codes from sklearn.model_selection.StratifiedKFold
unique_y, y_inversed = np.unique(y, return_inverse=True)
y_counts = np.bincount(y_inversed)
min_groups = np.min(y_counts)
```

[np.bincount](https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html)

## numpy.full¶

```
# example:
>>> np.full((2, 2), np.inf)
array([[ inf,  inf],
       [ inf,  inf]])
>>> np.full((2, 2), 10)
array([[10, 10],
      [10, 10]])

# source codes form sklearn.model_selection.KFold
>>> fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
array([92, 92, 92, 92, 92, 92, 92, 92, 92, 92])
>>> fold_sizes[:n_samples % n_splits] += 1
array([93, 93, 93, 93, 93, 93, 93, 93, 93, 92])
```
[numpy.full](https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html)

## np.in1d & np.isin
> We recommend using isin instead of in1d for new code.
```
In [183]: x,y=np.unique(['a','b','c','a','a','b','b'],return_inverse=True)

In [184]: x
Out[184]: array(['a', 'b', 'c'], dtype='<U1')

In [185]: y
Out[185]: array([0, 1, 2, 0, 0, 1, 1])

In [186]: np.in1d(y,[0,1])
Out[186]: array([ True,  True, False,  True,  True,  True,  True])

In [187]: np.isin(y,[0,1])
Out[187]: array([ True,  True, False,  True,  True,  True,  True])

```
[scipy np.in1d](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.in1d.html)

## reshape
```
# vector to image
  return img.reshape((shape[1], shape[0])).T
```

```
In[7]: a=np.arange(10)
In[8]: a
Out[8]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
In[9]: a.reshape(2,-1)
Out[9]:
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
In[10]: a.reshape(-1,2)
Out[10]:
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
```
如果`a = np.array([x1,y1,x2,y2,x3,y3,...])`, `a.reshape(-1,2)`有奇效

NOTE flatten always return a copy
[ravel](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ravel.html)
[flatten vs. ravel vs. reshape](https://stackoverflow.com/questions/28930465/what-is-the-difference-between-flatten-and-ravel-functions-in-numpy)

## np.flatnonzero

> Return indices that are non-zero in the flattened version of a.

```
In [194]: np.flatnonzero(np.array([0,0,1,2,3,0]))
Out[194]: array([2, 3, 4])

In [195]: np.flatnonzero(np.array([[0,1,2],[2,3,4]]))
Out[195]: array([1, 2, 3, 4, 5])

```

## numpy.logical_not
```
>>> np.logical_not([True, False, 0, 1])
array([False,  True,  True, False])

In [85]: x
Out[85]: array([ 1,  2,  0,  0, -1, -2])

In [86]: np.invert(x)
Out[97]: array([-2, -3, -1, -1,  0,  1])

In [98]: ~x
Out[101]: array([-2, -3, -1, -1,  0,  1])

In [102]: np.logical_not(x)
Out[117]: array([False, False,  True,  True, False, False])
```
Note that `~` only works when `x`'s dtype is boolean
[stackoverflow: np.invert/~/np.logical_not](https://stackoverflow.com/questions/13728708/inverting-a-numpy-boolean-array-using/22225030)
[numpy.logical_not](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logical_not.html)

# spacemacs

##
[Pasting text into search after pressing "/" ](https://www.reddit.com/r/spacemacs/comments/4drxvv/pasting_text_into_search_after_pressing/)

## keymap

头疼的键冲突问题，对于键空间本就紧凑的原生 Emacs 而言简直就是一场灾难。不过好在解决的办法其实很简单：找一个没怎么用的 prefix 键作为专用的代理键，先映射到这个悬空的代理键上，然后再全局或者局部设置它。可以看下面的代码：
```
(define-key key-translation-map (kbd "A") (kbd "M-g A"))
(global/local-set-key (kbd "M-g A") 'your-command)
或者定义按键hong：
```
```
(defmacro m-map-key (obj key)
  `(let ((keystr (cadr ',key)) mapkey)
     (define-key key-translation-map ,key
       (if (not (symbolp ,obj)) ,obj
   (setq mapkey (kbd (concat "M-g " keystr)))
   (global-set-key mapkey ,obj) mapkey))))
```
[keymap](https://github.com/emacs-china/emacsist/blob/master/articles/2016-11-14%E9%82%A3%E5%B0%B1%E4%BB%8E%E5%A6%96%E8%89%B3%E9%85%B7%E7%82%AB%E7%9A%84%E5%BF%AB%E6%8D%B7%E9%94%AE%E5%BC%80%E5%A7%8B%E5%90%A7%EF%BC%81%EF%BC%88%E4%B8%80%EF%BC%89.org)

# vim 
## reg

`:[range]s/pattern/string/[c,e,g,i]`

## jedi && YCM
`YouCompleteMe` 和`jedi`冲突,导致在第一次函数提示时候总是卡很久，但是jedi具有其他优秀特性，比如参数提示；
YCM具有其他语言的提示能力,jedi提供了一个参数：
`let g:jedi#completions_enabled = 0 `
可以禁用jedi提示，但保留其它优秀性能！！！

## vimrc

- Tab
```
map <C-h> :tabprev<CR>
map <C-l> :tabnext<CR>
if !exists('g:lasttab')
  let g:lasttab = 1
endif
map <C-j> :exe "tabn ".g:lasttab<CR>
map <C-k> :exe "tabn ".g:lasttab<CR>
au TabLeave * let g:lasttab = tabpagenr()
```
在window之间切换可以使用Ctrl + w + h/j/k/l（向左、下、上、右切换）或者Ctrl + w + w（在窗口间切换）。
如果要调整window的大小，分窗口是水平分隔还是垂直分隔两种情况：
如果是水平分隔可以使用`:nwinc +/-`把当前激活窗口高度增加、减少n个字符高度，比如:`10winc +`
如果是垂直分隔可以使用`:nwinc >/<`把当前激活窗口宽度增加、减少n个字符宽度，比如:`5winc >`

## word/sentence/block etc操作和选定
[vim caw/daw](https://stackoverflow.com/questions/7267375/what-does-vaw-mean-in-vim-in-normal-mode-and-also-caw-and-daw)

## mark
[vim技巧: vim标记(Mark)](https://www.jianshu.com/p/37538ec6d8f7)

## surround/repeat
[vim插件: surround & repeat[成对符号编辑]](http://www.wklken.me/posts/2015/06/13/vim-plugin-surround-repeat.html)

## 列操作
可以进行批量注释（在多列最前端插入注释符号)
[vim块操作：列删除、列插入](https://blog.csdn.net/MrJonathan/article/details/51887980)

## ctrlp 快速搜索/打开文件
http://www.wklken.me/posts/2015/06/07/vim-plugin-ctrlp.html

## gundo
```
try
    set undodir=~/.vim/temp_dirs/undodir
    set undofile
    catch
endtry
```
[gundo 展示](http://www.wklken.me/posts/2015/06/13/vim-plugin-gundo.html)
[参考博客及永久保存配置](http://foocoder.com/2014/04/15/mei-ri-vimcha-jian-vim-che-xiao-shu-gundo-dot-vim/)

## 寄存器
`:reg`
[参考博客1](http://liuzhijun.iteye.com/blog/1830931)
[参考博客2](https://harttle.land/2016/07/25/vim-registers.html)
[一定要有+clipboard](https://www.zhihu.com/question/19863631)

## gitgutter
[gitgutter 中文简介博客](http://foocoder.com/2014/04/21/mei-ri-vimcha-jian-xian-shi-git-diff-gitgutter-dot-vim/)
[github gitgutter](https://github.com/airblade/vim-gitgutter)

# cv
## github发现的一个图像增强lib
[github albumentations](https://github.com/albu/albumentations)


# grep,sed,awk
http://blog.51cto.com/lq2419/1238880

# plt style
https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html

# sns
## distplot
https://seaborn.pydata.org/generated/seaborn.distplot.html

# click
[命令行神奇Click](http://funhacks.net/2016/12/20/click/)
[click](https://click.palletsprojects.com/en/7.x/)

# python

## X/X_/_X
[python命名规范](https://www.jianshu.com/p/a793c0d960fe)

## __iter__/__get_item
[python魔法方法](https://pycoders-weekly-chinese.readthedocs.io/en/latest/issue6/a-guide-to-pythons-magic-methods.html)
[python魔法](http://wiki.jikexueyuan.com/project/explore-python/Class/magic_method.html)
[自定义迭代](https://blog.csdn.net/heyijia0327/article/details/45101639)

## abstractmethod,staticmethod,classmethod
[非常好的参考博客](https://foofish.net/guide-python-static-class-abstract-methods.html)
[参考博客2](https://mozillazg.com/2014/06/python-define-abstract-base-classes.html)

## moudle
https://docs.python.org/2/tutorial/modules.html


# DL svd通道裁剪
1. [svd相关](https://www.leiphone.com/news/201802/tSRogb7n8SFAQ6Yj.html)
说白了就是某一层的权值矩阵A经过SVD分解为U*[sigma,0]*V_H时候，U取m*k,右侧乘起来为k*n,结果就是m*n->(m+n)*k
前向号理解，A就直接用分解后的矩阵代替运算即可（不过貌似多了一次矩阵乘法）
疑问就是：参数量减少了,但反向咋做(我猜应该是先训好，后压缩只做前向吧）
2. [知乎各种压缩介绍2017](https://zhuanlan.zhihu.com/p/25797790)
3. [链接2中评论区附带推文，值得参考](https://zhuanlan.zhihu.com/p/26528060)

# 图像分割loss
1. 以前作天池大数据平台的肺癌检测题目，一开源unet采用的是dice loss
2. kaggle slat identification比赛看见kernel unet采用binary_cross_entropy loss
- 参考
[知乎回答](https://www.zhihu.com/question/264537057)

# jupyter notebook
```
@ipy.interact(idx = ipy.IntSlider(min=0,max=len(train_img_filepaths)),value=10,step=1)
```

# YAML
kaggle TGS-SaltDetection open-solution
```
import yaml
from attrdict import AttrDict
def read_yaml(filename):
    with open(filename) as f:
        config = yaml.load(f)
    return AttrDict(config)
```
yaml中，双引号支持转义字符（单引号为单纯的str)
如：`"\t"`表示tab
- 参考
[简书简单参考](https://www.jianshu.com/p/f21b9306a68d)
[阮一峰的网络日志](http://www.ruanyifeng.com/blog/2016/07/yaml.html)
[CSDN blog](https://blog.csdn.net/lmj19851117/article/details/78843486)
[官网](http://yaml.org/)
[pyyaml](https://pyyaml.org/wiki/PyYAMLDocumentation)
[c++实现](https://github.com/jbeder/yaml-cpp/wiki/Tutorial)

# 天气爬虫
- 参考
[历史天气](http://lishi.tianqi.com/chengdu/201807.html)
[CSDN python code](https://blog.csdn.net/haha_point/article/details/77197230)

# log & warn
## warnings
作用同logger.warn()类似
```
import warnings
warnings.warn('test')
```

## logging
```
import logging
def init_logger():
    logger = logging.getLogger('mylog')
    # 设置显示级别
    logger.setLevel(logging.INFO)
    # console handler
    handler = logging.StreamHandler(sys.stdout)
    # 因为logging是树状
    handler.setLevel(logging.INFO)
    handler.setFormatter(fmt=logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s', datefmt='%Y-%m-%d %H-%M-%S'))
    logger.addHandler(handler)
    return logger
logger = init_logger()
logger.info('test')
```
- 参考
[logging 简书](https://www.jianshu.com/p/feb86c06c4f4)
[python3 logging](https://docs.python.org/3/library/logging.html)

# text to art character
示例：
```
 __    __  .___________. __   __          _______.
|  |  |  | |           ||  | |  |        /       |
|  |  |  | `---|  |----`|  | |  |       |   (----`
|  |  |  |     |  |     |  | |  |        \   \ 
|  `--'  |     |  |     |  | |  `----.----)   |
 \______/      |__|     |__| |_______|_______/

```
- 参考
[使用网址](http://www.patorjk.com/software/taag/#p=display&f=Star%20Wars&t=UTILS)

# git

## git mirror
```
git clone --bare git://github.com/username/project.git
cd project.git
git push --mirror git@gitcafe.com/username/newproject.git
```
[git仓库迁移](https://my.oschina.net/kind790/blog/510601)
[git push mirror](https://help.github.com/articles/duplicating-a-repository/)

## GitPython
```
from git import Repo
repo = Repo('...')
repo.branches
repo.tags
...
```
[GitPython Tutorial](https://gitpython.readthedocs.io/en/stable/tutorial.html)

# pytorch
## optimizer
```python
# params in Adam
# pytorch 0.3.1
def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
              weight_decay=0):
    defaults = dict(lr=lr, betas=betas, eps=eps,

weight_decay=weight_decay)
    super(Adam, self).__init__(params, defaults)
```
some source code in optimizer.py
note that `default is required` is always `False`, this may be the todo code.

```python
# pytorch 0.3.1
required = object()
for name, default in self.defaults.items():
    if default is required and name not in param_group:
        raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                          name)
    else:
        param_group.setdefault(name, default)
```

## *callbacks
- [ncullen93/torchsample](https://github.com/ncullen93/torchsample/blob/master/torchsample/callbacks.py)

## Delete layers in pretrained models
```python
class Identify(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    return x

```
- [How to delete layer in pretrained model?](https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648)

## ConvTranspose2d
> fractionally-strided convolutions (in some recent papers, these are wrongly called deconvolutions
[[Question] Transposed Conv equivalent to Upsampling + Conv? #7307](https://github.com/keras-team/keras/issues/7307)
[Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)
[An Introduction to different Types of Convolutions in Deep Learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
[Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic)

## ! onnx & dot
pytorch model可视化可以通过onnx，然后转化为dot格式！
```
python3 ~/workspace/onnx/onnx/tools/net_drawer.py --input resnet.onnx --output resnet.dot --embed_docstring
dot -Tsvg resnet.dot -o resnet.svg
```
直接打开svg图片即可.
pytorch export onnx时候，可能报错
- forward中出现reshape报错。修正为.view即可。
- pool的一个参数ceil_mode=True貌似也会报错（默认是False）
- 可视化的网络图中，自己实现的resnet在每个conv前面都比官方的多一个输入框，原因是：Conv2d的bias默认=True!!!

[pytorch model to onnx](https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html)
[onnx net drawer tool](https://github.com/onnx/tutorials/blob/master/tutorials/VisualizingAModel.md)

## in_place
[resnet relu & += inplace](https://github.com/pytorch/pytorch/issues/5687)

- official implementation, `relu` after `+=`
[torchvision implementation of resnet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L51-L52)

## torch.utils.data.DataLoader
Dataset可以用于遍历，但是缺乏：
*    Batching the data
*    Shuffling the data
*    Load the data in parallel using multiprocessing workers.
torch.utils.data.DataLoader is an iterator which provides all these features.

## torch.from_numpy(np)
pycharm不提示该函数，但是在ipython中提示！

## torchvision.utils.make_grid
4x3x32x32->3x32x138 也就是完成了多图拼接为一个横图，方便imshow等显示！
conda install pytorch torchvision -c pytorch

## learns
```
def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                return_indices=False, ceil_mode=False):
    super(_MaxPoolNd, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride or kernel_size
# 注意学习stride这个写法
```

## zero_grad()
```
 optimizer.zero_grad()
 net.zero_grad()
```
if optimizer = optim.Optimizer(net.parameters()),they are the same.
```
    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()
```

# graphviz
lightgbm和steppy中在可视化树图时都有用到该库！
在jupyter运行:
```
import graphviz
d = graphviz.Digraph()
d.edge('test','hello')
d.edge('test','world')
d
```
[graphvia docs](https://graphviz.readthedocs.io/en/stable/manual.html#attributes)
[绘图工具graphviz学习使用](http://hustlijian.github.io/tutorial/2015/05/29/graphviz-learn.html)

# image read speed
```
image = cv2.imread(...)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = image[:,:,::-1] # for better speed
```
[知乎问答](https://www.zhihu.com/question/48762352)
[知乎软文,留意讨论区](https://zhuanlan.zhihu.com/p/30383580)

# opencv

## cv2.resize()
```
cv2.resize(image, (cols, rows))
# 即 w，h排序。区别于skimage transform.resize
```

# linux
## shuf
shuffle lines
```
用法： shuf [选项]... [文件]
　或者:  shuf -e [选项]... [参数]...
　或者:  shuf -i LO-HI [选项]...
Write a random permutation of the input lines to standard output.

Mandatory arguments to long options are mandatory for short options too.
  -e, --echo                treat each ARG as an input line
  -i, --input-range=LO-HI   treat each number LO through HI as an input line
  -n, --head-count=COUNT    output at most COUNT lines
  -o, --output=FILE         write result to FILE instead of standard output
      --random-source=FILE  get random bytes from FILE
  -r, --repeat              output lines can be repeated
  -z, --zero-terminated     end lines with 0 byte, not newline
      --help		显示此帮助信息并退出
      --version		显示版本信息并退出

如果没有指定文件，或者文件为"-"，则从标准输入读取。

GNU coreutils online help: <http://www.gnu.org/software/coreutils/>
请向<http://translationproject.org/team/zh_CN.html> 报告shuf 的翻译错误
要获取完整文档，请运行：info coreutils 'shuf invocation'

```

## wc
word/line count
```
用法：wc [选项]... [文件]...
　或：wc [选项]... --files0-from=F
Print newline, word, and byte counts for each FILE, and a total line if
more than one FILE is specified.  With no FILE, or when FILE is -,
read standard input.  A word is a non-zero-length sequence of characters
delimited by white space.
The options below may be used to select which counts are printed, always in
the following order: newline, word, character, byte, maximum line length.
  -c, --bytes            print the byte counts
  -m, --chars            print the character counts
  -l, --lines            print the newline counts
      --files0-from=文件	从指定文件读取以NUL 终止的名称，如果该文件被
					指定为"-"则从标准输入读文件名
  -L, --max-line-length	显示最长行的长度
  -w, --words			显示单词计数
      --help		显示此帮助信息并退出
      --version		显示版本信息并退出
```

## awk
```
Usage: awk [POSIX or GNU style options] -f progfile [--] file ...
Usage: awk [POSIX or GNU style options] [--] 'program' file ...
POSIX options:		GNU long options: (standard)
	-f progfile		--file=progfile
	-F fs			--field-separator=fs
	-v var=val		--assign=var=val
Short options:		GNU long options: (extensions)
	-b			--characters-as-bytes
	-c			--traditional
	-C			--copyright
	-d[file]		--dump-variables[=file]
	-e 'program-text'	--source='program-text'
	-E file			--exec=file
	-g			--gen-pot
	-h			--help
	-L [fatal]		--lint[=fatal]
	-n			--non-decimal-data
	-N			--use-lc-numeric
	-O			--optimize
	-p[file]		--profile[=file]
	-P			--posix
	-r			--re-interval
	-S			--sandbox
	-t			--lint-old
	-V			--version

To report bugs, see node `Bugs' in `gawk.info', which is
section `Reporting Problems and Bugs' in the printed version.

gawk is a pattern scanning and processing language.
By default it reads standard input and writes standard output.

Examples:
	gawk '{ sum += $1 }; END { print sum }' file
	gawk -F: '{ print $1 }' /etc/passwd

```

# sklearn

## GroupShuffleSplit

- `sklearn.model_selection.StratifiedKFold¶` 分层抽样
- `sklearn.model_selection.GroupShuffleSplit¶` 分组抽样

各种cv对比！！
[cvs](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py)

## scaling
[注意各种归一化的区分](http://benalexkeen.com/feature-scaling-with-scikit-learn/)

## OneHotEncoder/LabelEncoder/LabelBinarizer/Binarizer
1. ohe需要元素都是数值，如果出现str会报错
2. le用于将str编码为int，但不做one hot
3. 所以需要两步：ohe->le
4. binarizer是将数值二值化；比自己手动二值好在fit时候保存了参数，而且能pipeline
5. lb用于list或者Series（1维）one-hot
```
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])

lb.classes_

lb.transform([1, 6])
```
[blog(这人踩了和我近乎相同的坑，包括顺序)](https://ask.hellobi.com/blog/DataMiner/4897)

## partial dependence plots
```
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    clf.fit(X_train, y_train)
    print(" done.")

    print('Convenience plot with ``partial_dependence_plots``')

    features = [0, 5, 1, 2, (5, 1)]
    fig, axs = plot_partial_dependence(clf, X_train, features,
                                       feature_names=names,
                                       n_jobs=3, grid_resolution=50)
```
个人理解：
1. `partial dependence plot`需要model和data才能绘制，直觉上，考虑feature a和target关系，
可以直接把a分成很多bins(如果a为category的就直接用其类别)，分bin统计target，画a-target的变化曲线，
就可以得到想要的结果。But！想象一个异或问题，如果单独统计x或者y都是一半一半，将的出target和x及y无关的结论。
导致此错误的根本原因是统计可用的假设是feature和target是线性的,而线性模型无法学出异或，于是直接统计是不行的。
2. PDP的基本原理是，如果要观察feature a，就固定b,c,d等feature，变化a，看model输出变化（当然这里的前提是
feature独立，由ALE等方法改进）。为何看model输出？和1中统计真值的target是两种操作！所以这种方式的假设是：
model fit的非常好，model非常可信！

> We don’t analyze data, we analyze models.
- 下面这文章太吊了，必看！！第五章还把LR讲得很明白！
[git-io: interpretability-importance](https://christophm.github.io/interpretable-ml-book/interpretability-importance.html#)
- [Random forest positive/negative feature importance](https://stats.stackexchange.com/questions/288736/random-forest-positive-negative-feature-importance)
- [Partial Dependence Plots](http://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html)
- [non-linear regression R-squre](http://statisticsbyjim.com/regression/r-squared-invalid-nonlinear-regression/)

SHAP
Interpretation lightgbm model based on SHAP.
Note that `shap` doesn't handle category features well
- [SHAP github](https://github.com/slundberg/shap)
- [Consistent Individualized Feature Attribution for Tree Ensembles](https://arxiv.org/abs/1802.03888)
- [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)


# icons website
[icons1](https://icons8.com/icon/set/basic-system/all)
[icons2](https://www.flaticon.com/packs/data-analytics)

# GAN

## Mode Collapse
[GAN](https://sinpycn.github.io/2017/05/10/GAN-Tutorial-Research-Frontiers.html)
[mode collapse in GANs](http://aiden.nibali.org/blog/2017-01-18-mode-collapse-gans/)

# LSTM
[lstm](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)


# TODO
1. lightgbm params will be changed!! PR

# DAILY
1. learn english at least 1 hour per day
2. conquer 2 leetcode easy problems
3. learn something new
4. read a kaggle kernel
