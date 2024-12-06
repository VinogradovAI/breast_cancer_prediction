
## **Проектное исследование на основе открытых данных в области биомедицины**


https://minds.wisconsin.edu/bitstream/handle/1793/59692/TR1131.pdf;jsessionid=944B13EC1D0D6DCB528D22A999948C8E?sequence=1
* ссылка на статью
## Извлечение ядерных признаков для диагностики опухоли молочной железы
У. Стрит, У. Вольберг, О. Мангасарян
Опубликовано в журнале «Электронная визуализация» 29 июля 1993 г.
#%% md
### **Аннотация**
Для создания высокоточной системы диагностики опухолей молочной железы были использованы интерактивные методы обработки изображений, а также индуктивный классификатор на основе линейного программирования. Небольшая часть препарата, полученного с помощью тонкоигольной аспирационной биопсии, отбирается и оцифровывается. Это позволяет проводить точный автоматизированный анализ размера, формы и текстуры ядер. Для каждого ядра вычисляются десять таких характеристик, и для диапазона изолированных клеток определяются среднее значение, наибольшее (или «наихудшее») значение и стандартная ошибка каждой характеристики. После анализа таким образом 569 изображений были протестированы различные комбинации характеристик, чтобы найти те, которые лучше всего отделяют доброкачественные образцы от злокачественных. Десятикратная точность перекрестной проверки в 97% была достигнута с использованием одной разделительной плоскости по трем из тридцати признаков: средней текстуре, наихудшей площади и наихудшей гладкости. Это представляет собой улучшение по сравнению с лучшими результатами диагностики в медицинской литературе. В настоящее время система используется в больницах Университета Висконсина. Тот же набор признаков также использовался в гораздо более сложной задаче прогнозирования отдаленного рецидива злокачественной опухоли у пациентов, что привело к точности 86%.
#%% md

## **Цель:**
Применить методы как классического, так и глубокого машинного обучения для достижения наилучших результатов в предсказании медицинских исходов на основе выбранного набора данных.

Для работы был использован датасет **Breast Cancer Wisconsin Dataset**, ссылка на репозиторий: (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

#%% md
### **Описание датасета: данные и признаки**

Датасет включает реальные данные пациенток, которым была проведена тонкоигольная аспирационная биопсия молочной железы для диагностики характера опухолевого образования.

Все признаки были получены в результате компьютерной обработки изображений биопсийного материала с использованием специализированного программного обеспечения. При этом анализировались только клеточные ядра, поэтому все **признаки представлены вычисленными программой специфическими характеристиками клеточных ядер**. (Nuclear feature extraction for breast tumor diagnosis By W. Street, W. Wolberg, O. Mangasarian. 1993 Published in Electronic imaging) (https://www.sci-hub.ru/10.1117/12.148698)

Ключевым этапом подготовки изображений для компьютерной обработки стало выделение границ клеточных ядер - прорисовку их контуров. На основе полученных линий контуров выполнялась дальнейшая обработка изображений и извлечение признаков.

Пример изображения - биоптата с выделенными программой границами клетчных ядер для дальнейшей обработки и извлечения признаков (Nuclear feature extraction for breast tumor diagnosis By W. Street, W. Wolberg, O. Mangasarian. 1993 Published in Electronic imaging, figure 2) (https://www.sci-hub.ru/10.1117/12.148698)

![Пример биоптата с выделенными программой ядрами клеток для дальнейшей обработки и извлечения признаков из статьи выше](attachment:image.png)




**Всего вычислялось 10 ключевых признаков:**

   1) *radius* (mean of distances from center to points on the perimeter)- среднее расстояние от центра ядра до его границы
   2) *texture* (standard deviation of gray-scale values) - определения разницы в интенсивности оттенков серого
   3) *perimeter* (длина "окружности" клеточного ядра)
   4) *area* (Площадь ядра измеряется простым подсчетом количества пикселей внутри выделенной границы и добавлением половины пикселей от периметра)
   5) *smoothness* (local variation in radius lengths) - локальные вариации длины радиуса клеточного ядра
   6) *compactness* (perimeter**2 / area - 1.0) - компактность (периметр^2 / площадь - 1,0)
   7) *concavity* (severity of concave portions of the contour) - степень вогнутости контура (линни границы) клеточного ядра
   8) *concave_points* (number of concave portions of the contour) - количество "вогнутостей"
   9) *symmetry* - симметрия клеточного ядер: находится самая длинная ось ядра, проходящая через центр, к ней проводят перпендикуляры и оценивают длину каждого из отрезков по обе стороны от оси.
   10) *fractal_dimension* ("coastline approximation" - 1) - фрактальная размерность («приближение береговой линии» - 1)

Все признаки численно моделировались таким образом, что бОльшие значения обычно указывают на более высокую вероятность злокачественности.

Для каждого признака вычислялось среднее значение, экстремальное (наибольшее) значение и стандартная ошибка, поэтому в финальном датасете для каждого изображения получалось 30 признаков.

Распределение по классам: 357 доброкачественных, 212 злокачественных


### **Методы исследования:**

1. **Предобработка данных:**
    - Изучение данных
    - Проверка на пропуски
    - Проверка на дубликаты
    - Проверка на наличие выбросов
    - Проверка на корреляцию признаков
    - Приведение данных к числовому виду
    - Нормализация данных
    - Разделение данных на обучающую и тестовую выборки
    - Построение графиков для визуализации данных
    - Проверка на баланс классов
    - PCA для снижения размерности
    - Модель эллипсоида для определения выбросов
    - и другие методы


**Ключевой момент**: ***При помощи различных подходов и их комбинаций были созданы 3 датасета с признаками для машинного обучения***

**Результаты предобработки:**
- Датасеты находятся в папке `data/processed`
- EDA в Jupiter Notebook-е 'notebooks/data_preprocessing_EDA'
- Индивидуальные работы с различными подходами - в папке `notebooks/personal_notebooks`


2. **Моделирование:**
    - Обучение моделей на основе классических методов машинного обучения
    - Обучение моделей на основе глубокого обучения
    - Сравнение результатов внутри одного и на различных датасетах
    - После кросс-валидации модели оцениваются на валидационной выборке
    - Подбор гиперпараметров
    - Тестовая оценка качества моделей используется после подбора гиперпараметров

Алгоритмы моделирования находятся в файле **breast_cancer_detection.py**

3. **Модели**:
    - Логистическая регрессия
    - Метод опорных векторов
    - Случайный лес
    - Градиентный бустинг
    - Нейронные сети
    - и другие

- Результаты моделирования находятся в папке `results`
- Индивидуальные работы с различными алгоритмами - в папке `notebooks/personal_notebooks`



## **Результаты:**

Из всех датасетов наилучшим образом показал себя простой двухкомпонентный подход:
- **Логарифмирование** данных
- **PCA** для снижения размерности с визуальным определением оптимального числа компонент (6)

Наилучшие результаты на валидационной выборке показала модель **XGBoost** с результатами:
- **Accuracy**: 0.97
- **F1-score**: 0.96

На тестовой выборке результаты были следующие:
- **Accuracy**: 0.99
- **F1-score**: 0.99

Сравнительные результаты моделей представлены в папке 'results/tables'
Графики обучения и валидации моделей - в папке 'results/plots'
