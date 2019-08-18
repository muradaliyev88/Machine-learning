_Qərar ağacı sinifləndirici,cəlbedici modellərdir əgər şərh olunan modelə önəm veririksə.Çox mükəmməl alqoritmlərdir, mürəkkəb məlumat setini uyğunlaşdırmağı bacarırlar. Qərarı ağacını bir sıra suallar verərək məlumatlarımızı parçalayan bunlar əsasında qərar verərən alqoritm kimi düşünə bilərik. Qərar ağacları, bu gün mövcud olan ən güclü Maşın Öyrənmə alqoritmləri arasında olan Random Forests əsas komponentləridir. Belə modellərə tez-tez ağ qutu modelləri deyilir. Bunun əksinə olaraq, görəcəyimiz təsadüfi meşələr və ya neyron şəbəkələr ümumiyyətlə qara qutu modelləri hesab olunur._

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/Fig%201-18e1a01b.png)

_Qərar ağacları olduqca intuitivdir və qərarlarınıda şərh etmək asandır. Təlim setindəki özəlliklərə əsaslanaraq qərar ağacı modeli nümunələrin sinif etiketlərini  anlamaq üçün bir sıra sualları öyrənir.Yuxarıdakı şəkil kateqoriyalı dəyişənlərə əsaslanan qərar ağacı anlayışını təsvir edir._

_Qərar ağaclarını başa düşmək üçün, sadəcə birini quraq və onun necə proqnoz verdiyini nəzərdən keçirəkQərar ağaclarını başa düşmək üçün, sadəcə birini quraq və onun necə proqnoz verdiyini nəzərdən keçirək._

`from sklearn.datasets import load_iris`
`from sklearn.tree import DecisionTreeClassifier`
`from sklearn.model_selection import train_test_split`

`iris = load_iris()`

`X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=42)`
`tree = DecisionTreeClassifier(random_state=0)`
`tree.fit(X_train, y_train)`

<pre>DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')</pre>


`print("təlim setində doğruluq: %f" % tree.score(X_train, y_train))`
`print("test setində doğruluq: %f" % tree.score(X_test, y_test))`

<pre>təlim setində doğruluq: 1.000000
test setində doğruluq: 0.921053
</pre>

_Gördüyümüz kimi, təlim setindəki dəqiqlik 100%. Test setində dəqiqliyi təlim setindən 5% azdır.
Yaxşı bəs sualları özümüz tənzimləyə bilərikmi necə  edək ki ard arda çoxlu yox biz istədyimiz qədər suallar versin bir sözlə  sualların sayın özümüz təyin edək məhdudlaşdıraq?_**Bunun üçün biz `max_depth` istifadə edə bilərik**

`tree = DecisionTreeClassifier(max_depth=2, random_state=0)`
`tree.fit(X_train, y_train)`

<pre>DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
            splitter='best')</pre>


`print("təlim setində doğruluq: %f" % tree.score(X_train, y_train))`
`print("test setində doğruluq: %f" % tree.score(X_test, y_test))`

<pre>təlim setində doğruluq: 0.964286
test setində doğruluq: 0.921053
</pre>


**Əvvəlcə `max_depth=2`** _daha sonra 1 daha sonra isə 7 edin nəticələrin yaxşılaşması sualların artmasından asılıdır_
# Güclü, zəif tərəfləri və parametrləri











