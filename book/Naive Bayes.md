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
_Bu, əsasən mətinlərin idman, siyasət, texnologiya və s. Kateqoriyasına aid olub-olmaması üçün istifadə olunur._

### Bernoulli Naive Bayes:
_Bu multinomial naive bayesə bənzəyir, lakin proqnoz verənlər `boolean` dəyişkənlərdir. Sinif dəyişkənlərini tapmaq üçün istifadə etdiyimiz parametrlər yalnızca bəli və ya yox olur məsələn, mətndə bir söz meydana gəldi və ya gəlmədi_.

### Gaussian Naive Bayes:
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







