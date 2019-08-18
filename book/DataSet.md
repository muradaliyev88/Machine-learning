_**Maşın öyrənməsinə** başlamazdan əvvəl məlumat setiylə tanış olaq bunun üçün seaborn paketindən istifadə edəcəyik, burada fişerin irislərinin setindən isitifadə edəcəyik çox sadə setdi irislər haqqında ətraflı öyrənmək üçün bu_ [ linkdən ](https://en.wikipedia.org/wiki/Iris_flower_data_set)istifadə edə bilərsiniz daha sonra scikit-learnla tanış olacağıq.

` import seaborn as sns`

`import seaborn as sns`

`iris = sns.load_dataset('iris')`

`iris.head(10)`

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.6</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>


_Burada bir gülün müxtəlif ölçülərdə olan növləri göstərilir, burada matrisin sətirlərini nümunə olaraq (samples) və sətir sayısını (n-samples) olaraq ifadə edəcəyik.
Ümumi olaraq, matrisin sütunlarına özəlliklər olaraq baxacağıq ve sütunların sayısını (n-features) olarak göstərəcəyik._

_Tutaq ki, bir həvəskar bir adam topladığı gülün irisin hansı növü olduğunu bilmək istəyir. O, topladığı irislərin bəzi xüsusiyyətlərini  ölçdü.Bundan əlavə onları setosa, versicolor və virginica növlərinə aid etməsinə imkan verən irislərin məlumat setidə varıdı.Topladığı gülləri təxmin etmək üçün O iris növlərinin nümunələrinə baxaraq məsələni həll edəcək, bu cür həll ediləcən məsələyə müəllimlə təlim vermək deyilir. Bu məsələdə iris növlərindən birini təxmin etməliyik bunun üçün scikit-leaarndan istifadə edəcəyik._

**_Scikit-learn: Python proqramlaşdırma dili üçün ödənişsiz bir maşın öyrənmə kitabxanasıdır.Support vector machines, random forests, gradient boosting, k-means və DBSCAN da daxil olmaqla müxtəlif sinifləndirmə, reqressiya və kümələmə alqoritmləri mövcuddur._**

`from sklearn.datasets import load_iris`

`iris_dataset = load_iris()`

`iris_dataset.keys()`

<pre>dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])</pre>

`print(iris_dataset['DESCR'][:200] + "\n...") #bu məlumat setinin qısa bir açıqlamasıdır.`
<pre>.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive
...
</pre>

`iris_dataset['target_names'] # Proqnozlaşdırmaq istədiyimiz gülün növləri`
<pre>array(['setosa', 'versicolor', 'virginica'], dtype='&lt;U10')</pre>


`iris_dataset['feature_names'] #bu hər bir özəlliyin təsviri olan siyahı`
<pre>['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']</pre>


_Məlumatın özü `target` və `data` massivlərində qeyd olunur. **data** - çanağın uzunluğunun, çanağın genişliyinin, ləçəklərin uzunluğunun və ləçəklərin eni Numpy şəklində qeyd olunub. **target** - massivində güllərin çeşitləri qeyd olunub. `target` bir ölçülü bir sıra, hər bir çiçək üçün elementdir:_

`iris_dataset['data'][:10]`
<pre>array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2],
       [5.4, 3.9, 1.7, 0.4],
       [4.6, 3.4, 1.4, 0.3],
       [5. , 3.4, 1.5, 0.2],
       [4.4, 2.9, 1.4, 0.2],
       [4.9, 3.1, 1.5, 0.1]])</pre>

`iris_dataset['target'] # 0 – setosa,1 – versicolor, 2 – virginica`

<pre>array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])</pre>
