**Scikit-learn** _kitabxanasında, məlumat setini qarışdıran və onu iki hissəyə parçalayan_ `train_test_split`  _funksiyası var. Bu funksiya təlim setin məlumat etiketlərinin 75% -ni seçir , məlumatların qalan 25% -i test üçün seçir._ `train_test_split` _funksiyasını çağırmaq üçün:_

`from sklearn.model_selection import train_test_split`

**_İndi isə iris setimizi çağıraq_**

`from sklearn.datasets import load_iris`

`iris_dataset = load_iris()`

`X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)`

_İndi isə **KNeighborsClassifier** çağırmazdan əvvəl onun haqqında məlumat verim.._ __Sinifləndirilmədə istifadə olunan bu alqoritm sinifləndirilən özəlliklərdən daha çox hansına yaxın olmasıdı.__ 
_Yaxın qonşu (KNN) çox sadə, başa düşülən, çox yönlü və ən yaxşı maşın öyrənmə alqoritmlərindən biridir. KNN, maliyyə, səhiyyə, siyasi elm, əl yazısının aşkarlanması, görüntü tanıma və video tanıma kimi müxtəlif tətbiqlərdə istifadə olunur. Misal olaraq k=3 olsun  etiketlərdən 3 elelement alınır məsafə hesabına görə daha yaxın olan elementin sinifinə aid edilir._

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/KNN_final1_ibdm8a.webp)



`from sklearn.neighbors import KNeighborsClassifier #Biz burada KNN alqoritmin çağırdıq onu knn dəyişkəninə bağladıq`

`knn = KNeighborsClassifier(n_neighbors=5,metric="euclidean")`

Əgər biz **KNeighborsClassifier** parametrlərinə baxsaq aşağıdakı yazılanları görərik

`KNeighborsClassifier(
    ['n_neighbors=5', "weights='uniform'", "algorithm='auto'", 'leaf_size=30', 'p=2', "metric='minkowski'", 'metric_params=None', 'n_jobs=None', '**kwargs'],
)`

Əgər parametrləri dəyişdirməsək o bu dəyərlərlə işləyəyəcək. Burada `n_neighbors=5` yəni tapmaq istədiyimiz məlumat ona yaxın neçə qonşuyla aralarındakı məsafə ölçülsün. Parametrlərdən `metric='minkowski` haqqında biraz danışmaq istəyirəm digərləri haqqında özünüz maraqlanıb baxa bilərsiz.
### Minkowski məsafəsi

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/1_wWdhIZJd6y_v4C3Ze2tqFQ.png)

**Minkowski** məsafəsi ümumiləşdirilmiş uzağlıq ölçüsüdür. Burada ümumiləşdirilmiş, iki nöqtə arasındakı məsafəni müxtəlif yollarla hesablamaq üçün yuxarıdakı formulu dəyiştirə biləcəyik. p dəyərini dəyiştirə bilər və məsafəni üç müxtəlif yolla hesablaya bilərik.
> p = 1, Manhattan Distance

> p = 2, Euclidean Distance

> p = ∞, Chebychev Distance

İki nöqtə arasındakı məsafəni hesablamağımız lazımdırsa, **Manhetten** məsafəsini istifadə edirik. Yuxarıda qeyd edildiyi kimi, P dəyərini 1 olaraq təyin edərək **Manhetten** məsafəsini tapmaq üçün **Minkowski** məsafə düsturundan istifadə edirik.Deyək ki, iki nöqtə arasındakı məsafəni, d- x və y ilə hesablamaq istəyirik.

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/1_7NHkUCylraQu2H-S5N1nhA.png)

**D** məsafəsi aşağıdakı mütləq fərqdən istifadə edilərək hesablanacaq:

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/1_1pvCYxUipB2rK05cnU7_XQ.png)

burada, n- dəyişkənlərin sayı, *xi və yi*, iki ölçülü vektor fəzasında müvafiq olaraq *x və y* vektorlarının dəyişənləridir. yəni x = (x1, x2, x3, ...) və y = (y1, y2, y3, ...). d məsafə belə hesablanacaqdır

(x1 - y1) + (x2 - y2) + (x3 - y3) + … + (xn - yn).


### Euclidean Distance:
**Evklid** məsafəsi ən çox istifadə olunan məsafə ölçüsüdür. *P*-in dəyərini 2-yə qoyaraq **Minkowski** məsafə düsturu ilə hesablanır. Bu məsafə *'d'* formulunu aşağıdakı kimi yeniləyəcəkdir.**Evklid** məsafəsi düsturu bir müstəvidə iki nöqtə arasındakı məsafəni hesablamaq üçün istifadə edilir.

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/1_n6kmkzjKVTOWeXDxsx2daQ.png)

**Kiçik bir haşiyədən sonra qayıdıram kodlamaya))**

`knn.fit(X_train, y_train)`

<pre>KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform')</pre>


`fit()` metodu **knn** obyektinin özünü qaytarır (və dəyişdirir).İndi təxminimizi ala bilərik.Təsəvvür edin biz təbiətdə belə bir `[5, 2.9, 1, 0.2]` ölçülərə malik bir dənə **iris tapmışıq**. Təxmin olunanı tapmaq üçün `predict()` metodunu çağırırıq.

`import numpy as np`

`X_new = np.array([[5, 2.9, 1, 0.2]])`

`prediction = knn.predict(X_new)`

`print("Təxmin olundu: {}".format(iris_dataset['target_names'][prediction]))`

<pre>Təxmin olundu: ['setosa']
</pre>

Bundan əlavə, test etdiyimiz *modelin düzgünlüyünü hesablayan* `knn.score()` metodundan istifadə edə bilərik

`knn.score(X_test, y_test)`

<pre>0.9736842105263158</pre>

97% test setinin düzgünlüyünü göstərir.

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/rrr.jpg)

# Həqiqətəndə görəsən arxa planda K-NN alqoritmi necə işləyir???
**Bir neçə gün axtarıb araşdırdım Googledən soruşdum bir iki kitab oxuyandan sonra isə ortalığa belə bir kodlar dökülməyə başladı əvvəlcə** `dot()` **funksiyası daha sonra** `distance()` **funksiyaların yazdım**.**Distans funksiyası haqqında danışmışdıq onu kod halına bu cür gətirdik** `dot()` **funksiyası haqqında**
[bura baxa bilərsiz](https://github.com/muradaliyev88/Machine-learning/blob/master/backend/dot.ipynb)


`def dot_sca(v, w):`
    
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

`def dot_2d(v,w):`
    
    return v@w

`def dot(v, w):`
        
    if len(v.shape) !=1 and len(w.shape) !=1:
        f = np.array([dot_2d(v[i],w) for i in range(len(v))])        
        return f    
            
    else:
        print(dot_sca(v,w))


`def distance(v, w):`
    
    if len(v.shape) !=1 and len(w.shape) !=1:
        f = np.array([v_n - w_n for v_n,w_n in zip(v,w)])        
        d = dot(f,f)
        return repr(np.sqrt(d))
            
    else:
        g = np.array([v_n - w_n for v_n,w_n in zip(v,w)])
        return repr(np.sqrt(dot_sca(g,g)))
    
`def fit(X_train, y_train):` 
   
    return

`def k_nearest_neighbor(X_train, y_train, X_test, k):`   
 
    fit(X_train, y_train)    

    predictions = []

    for i in range(len(X_test)):
        predictions.append(predict(k,X_train,X_test[i, :], y_train  ))
    return np.asarray(predictions)    
    
`from collections import Counter`

`def predict(k, data, test,label):`

    distances = []
    targets = []  
    for i in range(len(data)):
        dist = (distance(test,data[i,:]))       
        distances.append((dist,i))
    distances = sorted(distances)
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])    
    return Counter(targets).most_common(1)[0][0]

`predict(1,X_train,X_test ,y_train)`
`predictions = k_nearest_neighbor(X_train, y_train, X_test, 1)`
`print(predictions)`

`from sklearn.metrics import accuracy_score`

`accuracy = accuracy_score(y_test, predictions)`
`print("test setinin düzgünlüyü {}".format(100*accuracy))`

<pre>[2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
 2]
test setinin düzgünlüyü 97.36842105263158
</pre>
