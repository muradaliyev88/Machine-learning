# Naive Bayes


**Naive Bayes Sinifləndirici adını İngilis riyaziyyatçı Tomas Bayes'dən (1701 - 7 Nisan 1761) alır.Naive Bayes, müəyyən bir sinifə aid bir məlumat nöqtəsinin hansı sinifə aid olma  ehtimalını hesablayan proqnozlar verən bir modelidir.Maşın öyrənməsində Naive Bayes sinifləndirici, Bayes teoremini özəlliklər arasında müstəqil fərziyyələri tətbiq etməyə əsaslanan siniflərə aid ehtimallar hesablayan  təlimat öyrənmə alt sinifindədir.Sinifləndirmə,özəlliklər və sinif ehtimallarını nəzərə alaraq ən çox ehtimal olunan sinif seçərək aparılır.**

_Bayes teoremindən istifadə edərək B-nin meydana gəldiyini nəzərə alsaq A olma ehtimalını tapa bilərik. Burada B dəlil, A isə hipotezdir. Burada edilən ehtimal, proqnozlaşdırıcıların / xüsusiyyətlərin müstəqil olmasıdır. Yəni müəyyən bir xüsusiyyətin olması digərinə təsir etmir. Buna görə sadə adlanır._
## $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$


>P(A|B):B nin ehtimalı verildikdə A hadisəsinin olma ehtimalı

>P(B|A):A nın ehtimalı verildikdə B hadisəsinin olma ehtimalı

>P(A):A hadisəsinin olma ehtimalı

>P(B):B hadisəsinin olma ehtimalı

__Bir hadisənin baş verməsi birdən çox hadisənin mövcudluğuna bağlı ola bilər. Məsələn, əlimizdə $X$ üçün birdən çox hadisələrimiz var, $y_1$ hadisəsi, $y_2$ hadisəsi   Belə güman edək ki, bu hadisələr bir-birindən asılı olmayaraq baş verir.  Bu vəziyyətdə,  $Y_1$ və $Y_2$ hadisələrinin reallaşdığı zaman $X$ hadisəsinin baş vermə ehtimalı, $Y_1$ hadisəsinin baş verdiyi zaman $X$ hadisəsinin baş vermə ehtimalını, $Y_2$ hadisəsinin baş verdiyi zaman $X$ hadisəsinin baş vermə ehtimalını və $X$ hadisəsinin tək baş vermə ehtimalını vurma yolu ilə əldə edilir.__


$P(X|Y_i,…, Y_n) = P(X). P(Y_1, X) … P(Y_n, X)$

$P(X|Y_i,…, Y_n) = P(X)\displaystyle\prod_{i=1}^{n} P(Y_i, X)$ 

_İki etiket arasında ehtimalı istənilən verilənin ehtimalını hesablmaq istəyiriksə əvvəlcə siniflərə $Y_1$ və $Y_2$ adlandıraq daha sonra hər etiket üçün sonrakı ehtimalların nisbətini hesablamaq lazımdır:_

## $\frac{P(Y_1 |özəlliklər)(P|Y_1)}{P(Y_1 |özəlliklər)(P|Y_1)}=\frac{P(özəlliklər |Y_1) P(Y_1)}{P(özəlliklər  | Y_2)P(Y_2)}$

_Hər bir sinif üçün bu generativ model ilə hər hansı bir məlumat nöqtəsi üçün $P$ xüsusiyyətlərinin $Y_1$ ehtimalını hesablamaq üçün sadə bir tarifimiz var və buna görə sonra gələnin nisbətini tez bir şəkildə hesablaya bilərik və hər hansı bir nöqtənin hansı etiketə uyğun olacağını müəyyənləşdirə bilərik._

Hava məlumatlarını burada istifadə edirik







# Naive Bayes Sinifləndiricinin növləri:

### Multinomial Naive Bayes:

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/introduction-to-text-classification-using-naive-bayes-19-638.jpg)

_Bu, əsasən mətinlərin idman, siyasət, texnologiya və s. Kateqoriyasına aid olub-olmaması üçün istifadə olunur._

### Bernoulli Naive Bayes:

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/f6a27bc34d0c2ab16cb8ba203e5348741b5521e6.png)

_Bu multinomial naive bayesə bənzəyir, lakin proqnoz verənlər `boolean` dəyişkənlərdir. Sinif dəyişkənlərini tapmaq üçün istifadə etdiyimiz parametrlər yalnızca bəli və ya yox olur məsələn, mətndə bir söz meydana gəldi və ya gəlmədi_.

### Gaussian Naive Bayes:

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/1_0If5Mey7FnW_RktMM5BkaQ.png)

_Proqnoz verənlər davamlı bir dəyəri aldığında və diskret olmadıqda, bu dəyərlərin gaussian distributordan nümunələndiyini varsayırıq._

_Naive Bayes alqoritmləri əsasən həssaslıq, spam filtirlənməsində, tövsiyə sistemlərində və s. istifadə olunur. Təxminlər sürətlə həyata keçməkdədir, lakin ən böyük çatışmazlığı proqnozçıların sərbəst olmamasıdır. Real həyatda əksər hallarda, proqnozlaşdırıcılar asılıdır, bu təsnifatın işinə mane olur._

_Keçən dəfə ki bəhsimizdə olduğu kimi naive bayes modelimizi əvvəlcə çağıracağıq daha sonra x,y təyin edəcəyik modeli uyğunlaşdırdıqdan sonra test üçün məlumatımzı daxil edib ehtimalmızı tapacağıq._

`from sklearn.naive_bayes import GaussianNB `

`clf = GaussianNB()`

`from sklearn.datasets import load_iris`

`iris_dataset = load_iris()`

`from sklearn.model_selection import train_test_split`

`X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)`

`clf.fit(X_train, y_train)`

<pre>GaussianNB(priors=None, var_smoothing=1e-09)</pre>

`import numpy as np`

`X_new = np.array([[5, 2.9, 1, 0.2]])`

`prediction = clf.predict(X_new)`

`print("Təxmin olundu: {}".format(iris_dataset['target_names'][prediction]))`


<pre>Təxmin olundu: ['setosa']
</pre>

`clf.score(X_test, y_test)`

<pre>1.0</pre>

_Ümumi olaraq bunları demək istəyirəmki məlumat toplusunda etiketlərin hansı siniflərə mənsub olması məlumdur və təsnifatlandırılacaq etiket üçün hər bir sinifin reallığa oxşarlığı proqnozlaşdırılır. Bu metodda etiketlərin statistik asılı olmaması fərz olunur.Ona görə sadədir ki təlim məlumatlarını yalnız bir dəfə keçmək yetərlidir və sadə əlaqələr yaranır._


<tbody><tr><td style="width: 96px ; text-align: center"><strong>Name</strong></td><td style="width: 96px ; text-align: center"><strong>Yellow</strong></td><td style="width: 96px ; text-align: center"><strong>Sweet</strong></td><td style="width: 96px ; text-align: center"><strong>Long</strong></td><td style="width: 97px ; text-align: center"><strong>Total</strong></td></tr><tr><td style="width: 96px ; text-align: center"><strong>Mango</strong></td><td style="width: 96px ; text-align: center">350</td><td style="width: 96px ; text-align: center">450</td><td style="width: 96px ; text-align: center">0</td><td style="width: 97px ; text-align: center">650</td></tr><tr><td style="width: 96px ; text-align: center"><strong>Banana</strong></td><td style="width: 96px ; text-align: center">400</td><td style="width: 96px ; text-align: center">300</td><td style="width: 96px ; text-align: center">350</td><td style="width: 97px ; text-align: center">400</td></tr><tr><td style="width: 96px ; text-align: center"><strong>Others</strong></td><td style="width: 96px ; text-align: center">50</td><td style="width: 96px ; text-align: center">100</td><td style="width: 96px ; text-align: center">50</td><td style="width: 97px ; text-align: center">150</td></tr><tr><td style="width: 96px ; text-align: center"><strong>Total</strong></td><td style="width: 96px ; text-align: center">800</td><td style="width: 96px ; text-align: center">850</td><td style="width: 96px ; text-align: center">400</td><td style="width: 97px ; text-align: center">1200</td></tr></tbody>


__Yuxarıdakı cədvələ əsasən 1200 meyvədən 650-si manqo, 400-ü banan, 150-i isə digərləri.650 manqonun cəmi 350-si  sarı.800 meyvə sarı, 850-si şirin, 400-ü isə uzun.__


<tbody><tr><td style="width: 96px ; text-align: center"><strong>Name</strong></td><td style="width: 96px ; text-align: center"><strong>Yellow</strong></td><td style="width: 96px ; text-align: center"><strong>Sweet</strong></td><td style="width: 96px ; text-align: center"><strong>Long</strong></td><td style="width: 97px ; text-align: center"><strong>Total</strong></td></tr><tr><td style="width: 96px ; text-align: center"><strong>Mango</strong></td><td style="width: 96px ; text-align: center">350/800=P(Mango|Yellow)</td><td style="width: 96px ; text-align: center">450/850</td><td style="width: 96px ; text-align: center">0/400</td><td style="width: 97px ; text-align: center">650/1200=P(Mango)</td></tr><tr><td style="width: 96px ; text-align: center"><strong>Banana</strong></td><td style="width: 96px ; text-align: center">400/800</td><td style="width: 96px ; text-align: center">300/850</td><td style="width: 96px ; text-align: center">350/400</td><td style="width: 97px ; text-align: center">400/1200</td></tr><tr><td style="width: 96px ; text-align: center"><strong>Others</strong></td><td style="width: 96px ; text-align: center">50/800</td><td style="width: 96px ; text-align: center">100/850</td><td style="width: 96px ; text-align: center">50/400</td><td style="width: 97px ; text-align: center">150/1200</td></tr><tr><td style="width: 96px ; text-align: center"><strong>Total</strong></td><td style="width: 96px ; text-align: center">800=P(Yellow)</td><td style="width: 96px ; text-align: center">850</td><td style="width: 96px ; text-align: center">400</td><td style="width: 97px ; text-align: center">1200</td></tr></tbody>



![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/1-1.png)

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/2-1.png)

![](https://github.com/muradaliyev88/Machine-learning/blob/master/images/3-1.png)

__Bizim nümunəmizdə, ən çox ehtimal sinif bananına görədir, uzun, şirin və sarı olan meyvə Naive Bayes Alqoritmi tərəfindən əldə edilən bir banandır.__


_Verilənlərinizdə davamlı bir dəyişən varsa, onda Multinomial və Bernoulli uyğun deyil. Bunu məhdudlaşdıra bilərsiniz, lakin daha yaxşı olar ki Gaussian istifadə edək: Aşağıda olan kodlar (`GaussianNB2`) naive bayes üçün yazılmış kodlardır._


`class GaussianNB2(object):`

    def __init__(self):
        pass

    def fit(self, X, y):
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
                    for i in separated])
        return self

    def _prob(self, x, mean, std):
        exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
        return np.log(exponent / (np.sqrt(2 * np.pi) * std))

    def predict_log_proba(self, X):
        return [[sum(self._prob(i, *s) for s, i in zip(summaries, x))
                for summaries in self.model] for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)



`nb = GaussianNB2()`

`nb.fit(X_train, y_train)`

`X_new = np.array([[5, 2.9, 1, 0.2]])`

`prediction = nb.predict(X_new)`

`print("Təxmin olundu: {}".format(iris_dataset['target_names'][prediction]))`

`print(nb.score(X_test, y_test))`


<pre>Təxmin olundu: ['setosa']
0.9736842105263158
</pre>


















