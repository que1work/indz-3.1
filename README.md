# 10 алгоритмів регресійного аналізу для прогнозування цін на будинки

## Зміст
1. [Вступ](#вступ)
2. [Імпорт необхідних бібліотек](#імпорт-необхідних-бібліотек)
3. [Завантаження та огляд даних](#завантаження-та-огляд-даних)
4. [Попередня обробка даних](#попередня-обробка-даних)
5. [Підготовка даних для моделювання](#підготовка-даних-для-моделювання)
6. [Реалізація алгоритмів регресії](#реалізація-алгоритмів-регресії)
   * [1. Лінійна регресія](#1-лінійна-регресія)
   * [2. Робастна регресія](#2-робастна-регресія)
   * [3. Ridge регресія](#3-ridge-регресія)
   * [4. LASSO регресія](#4-lasso-регресія)
   * [5. Elastic Net регресія](#5-elastic-net-регресія)
   * [6. Поліноміальна регресія](#6-поліноміальна-регресія)
   * [7. Стохастичний градієнтний спуск](#7-стохастичний-градієнтний-спуск)
   * [8. Штучні нейронні мережі](#8-штучні-нейронні-мережі)
   * [9. Random Forest регресор](#9-random-forest-регресор)
   * [10. Метод опорних векторів](#10-метод-опорних-векторів)
7. [Порівняння результатів моделей](#порівняння-результатів-моделей)
8. [Висновки](#висновки)

## Вступ

У цьому блокноті ми розглянемо застосування 10 різних алгоритмів регресійного аналізу для прогнозування цін на будинки, використовуючи датасет "House Prices: Advanced Regression Techniques" з платформи Kaggle. Метою є не тільки реалізація цих алгоритмів, а й розуміння їх особливостей, переваг та недоліків, а також інтерпретація отриманих результатів.

## Імпорт необхідних бібліотек

```python
# Імпорт базових бібліотек для аналізу даних
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Бібліотеки для обробки даних
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

# Бібліотеки для реалізації регресійних алгоритмів
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
```

## Завантаження та огляд даних

```python
# Завантаження навчальних та тестових даних
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

# Відображення розміру даних
print("Розмір навчального набору:", train_df.shape)
print("Розмір тестового набору:", test_df.shape)

# Перегляд перших 5 рядків даних
train_df.head()
```

**Результат виконання:**
```
Розмір навчального набору: (1460, 81)
Розмір тестового набору: (1459, 80)
```

Дані містять 1460 зразків будинків з 81 ознакою у навчальному наборі та 1459 зразків з 80 ознаками у тестовому (без цільової змінної SalePrice).

```python
# Перевірка відсутніх значень
missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
print("Кількість відсутніх значень у кожній колонці:")
print(missing_values)

# Статистика для цільової змінної
print("Статистика для цільової змінної (SalePrice):")
print(train_df['SalePrice'].describe())

# Дослідження цільової змінної (SalePrice)
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('Розподіл цін на будинки')
plt.xlabel('Ціна')
plt.ylabel('Частота')
plt.show()

# Перевірка нормальності розподілу цільової змінної
plt.figure(figsize=(10, 6))
stats.probplot(train_df['SalePrice'], plot=plt)
plt.title('Q-Q графік для цін на будинки')
plt.show()
```

**Результат виконання:**
```
Кількість відсутніх значень у кожній колонці:
PoolQC          1453
MiscFeature     1406
Alley           1369
Fence           1179
MasVnrType       872
FireplaceQu      690
LotFrontage      259
GarageType        81
GarageYrBlt       81
GarageFinish      81
dtype: int64

Статистика для цільової змінної (SalePrice):
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
Name: SalePrice, dtype: float64
```

**[МІСЦЕ ДЛЯ ГРАФІКІВ: гістограма розподілу цін на будинки та Q-Q графік]**

Як видно з гістограми, розподіл цін на будинки має позитивну асиметрію (правосторонній хвіст). Q-Q графік також показує відхилення від нормальності розподілу. Тому для покращення результатів моделювання доцільно застосувати логарифмічне перетворення до цільової змінної.

## Попередня обробка даних

```python
# Логарифмічне перетворення цільової змінної
train_df['SalePrice_log'] = np.log1p(train_df['SalePrice'])

# Перевірка розподілу після перетворення
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice_log'], kde=True)
plt.title('Розподіл логарифму цін на будинки')
plt.xlabel('Log(Ціна+1)')
plt.ylabel('Частота')
plt.show()

# Перевірка нормальності після перетворення
plt.figure(figsize=(10, 6))
stats.probplot(train_df['SalePrice_log'], plot=plt)
plt.title('Q-Q графік для логарифму цін на будинки')
plt.show()

# Об'єднання навчальних і тестових даних для спільної обробки
all_data = pd.concat([train_df.drop(['SalePrice', 'SalePrice_log'], axis=1), test_df])
print("Розмір об'єднаних даних:", all_data.shape)

# Підрахунок загальної кількості пропущених значень
print("Загальна кількість пропущених значень:", all_data.isnull().sum().sum())

# Заповнення відсутніх значень
# Для числових стовпців використовуємо медіану
numeric_cols = all_data.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    all_data[col].fillna(all_data[col].median(), inplace=True)

# Для категоріальних стовпців використовуємо моду (найчастіше значення)
categorical_cols = all_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    all_data[col].fillna(all_data[col].mode()[0], inplace=True)

# Перевірка відсутніх значень після заповнення
print("Кількість пропущених значень після заповнення:", all_data.isnull().sum().sum())
```

**Результат виконання:**
```
Розмір об'єднаних даних: (2919, 80)
Загальна кількість пропущених значень: 15707
Кількість пропущених значень після заповнення: 0
```

**[МІСЦЕ ДЛЯ ГРАФІКІВ: гістограма та Q-Q графік після логарифмічного перетворення]**

Після логарифмічного перетворення розподіл цільової змінної став більш схожим на нормальний, що підтверджується Q-Q графіком.

## Підготовка даних для моделювання

```python
# Перетворення категоріальних змінних у числові (one-hot encoding)
all_data_encoded = pd.get_dummies(all_data, drop_first=True)
print("Розмір даних після кодування:", all_data_encoded.shape)

# Розділення даних назад на навчальні та тестові набори
X_train = all_data_encoded.iloc[:train_df.shape[0], :]
X_test = all_data_encoded.iloc[train_df.shape[0]:, :]

# Цільова змінна для навчання (використовуємо логарифмічне перетворення)
y_train = train_df['SalePrice_log']

# Розділення навчальних даних на тренувальний та валідаційний набори
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print("Розмір тренувальних даних:", X_train_split.shape)
print("Розмір валідаційних даних:", X_val.shape)

# Стандартизація числових ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)
```

**Результат виконання:**
```
Розмір даних після кодування: (2919, 245)
Розмір тренувальних даних: (1168, 245)
Розмір валідаційних даних: (292, 245)
```

## Реалізація алгоритмів регресії

### 1. Лінійна регресія

**Термін:** Лінійна регресія — це статистичний метод, який моделює лінійне співвідношення між залежною змінною і однією або кількома незалежними змінними.

```python
# Створення та навчання моделі лінійної регресії
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_split)

# Прогнозування на валідаційному наборі
lr_pred = lr_model.predict(X_val_scaled)

# Оцінка моделі
lr_mse = mean_squared_error(y_val, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_val, lr_pred)

print(f"Лінійна регресія:")
print(f"RMSE: {lr_rmse:.6f}")
print(f"R²: {lr_r2:.6f}")
```

**Результат виконання:**
```
Лінійна регресія:
RMSE: 0.215421
R²: 0.751322
```

**Інтерпретація:** Лінійна регресія показала задовільні результати з R² = 0.751322, що означає, що модель пояснює близько 75% варіації в логарифмічних цінах на будинки. RMSE = 0.215421 (у логарифмічній шкалі) вказує на середню помилку прогнозу.

### 2. Робастна регресія

**Термін:** Робастна регресія — це форма регресійного аналізу, яка розроблена для менш чутливої до викидів (аномальних значень) моделі за рахунок використання різних методів оцінки.

```python
# Створення та навчання моделі робастної регресії (RANSAC)
ransac = RANSACRegressor(
    LinearRegression(), 
    max_trials=100, 
    min_samples=50, 
    loss='absolute_error', 
    random_state=42
)
ransac.fit(X_train_scaled, y_train_split)

# Прогнозування на валідаційному наборі
ransac_pred = ransac.predict(X_val_scaled)

# Оцінка моделі
ransac_mse = mean_squared_error(y_val, ransac_pred)
ransac_rmse = np.sqrt(ransac_mse)
ransac_r2 = r2_score(y_val, ransac_pred)

print(f"Робастна регресія (RANSAC):")
print(f"RMSE: {ransac_rmse:.6f}")
print(f"R²: {ransac_r2:.6f}")
```

**Результат виконання:**
```
Робастна регресія (RANSAC):
RMSE: 0.136958
R²: 0.899484
```

**Інтерпретація:** Робастна регресія (RANSAC) показала значно кращі результати порівняно з простою лінійною регресією з R² = 0.899484 та RMSE = 0.136958. Це вказує на те, що в даних є викиди, які суттєво впливають на модель, і робастний метод ефективно справляється з ними.

### 3. Ridge регресія

**Термін:** Ridge регресія — це метод лінійної регресії з регуляризацією L2, який додає штраф до суми квадратів коефіцієнтів моделі, що допомагає запобігти перенавчанню.

```python
# Створення та навчання моделі Ridge регресії
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train_split)
    ridge_pred = ridge.predict(X_val_scaled)
    ridge_score = r2_score(y_val, ridge_pred)
    ridge_scores.append(ridge_score)

# Визначення найкращого параметра alpha
best_alpha_idx = np.argmax(ridge_scores)
best_alpha = alphas[best_alpha_idx]

# Навчання найкращої моделі
ridge_best = Ridge(alpha=best_alpha, random_state=42)
ridge_best.fit(X_train_scaled, y_train_split)
ridge_pred = ridge_best.predict(X_val_scaled)

# Оцінка моделі
ridge_mse = mean_squared_error(y_val, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_val, ridge_pred)

print(f"Ridge регресія (alpha={best_alpha}):")
print(f"RMSE: {ridge_rmse:.6f}")
print(f"R²: {ridge_r2:.6f}")
```

**Результат виконання:**
```
Ridge регресія (alpha=0.1):
RMSE: 0.152179
R²: 0.875900
```

**Інтерпретація:** Ridge регресія з оптимальним параметром регуляризації alpha = 0.1 показала хороші результати (R² = 0.875900, RMSE = 0.152179), хоча й трохи гірші, ніж робастна регресія. Регуляризація L2 допомагає зменшити перенавчання моделі.

### 4. LASSO регресія

**Термін:** LASSO (Least Absolute Shrinkage and Selection Operator) — це метод лінійної регресії з регуляризацією L1, який додає штраф до абсолютних значень коефіцієнтів, що може привести до виключення деяких ознак (встановлення їх коефіцієнтів на нуль).

```python
# Створення та навчання моделі LASSO регресії
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
lasso_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train_split)
    lasso_pred = lasso.predict(X_val_scaled)
    lasso_score = r2_score(y_val, lasso_pred)
    lasso_scores.append(lasso_score)

# Визначення найкращого параметра alpha
best_alpha_idx = np.argmax(lasso_scores)
best_alpha = alphas[best_alpha_idx]

# Навчання найкращої моделі
lasso_best = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
lasso_best.fit(X_train_scaled, y_train_split)
lasso_pred = lasso_best.predict(X_val_scaled)

# Оцінка моделі
lasso_mse = mean_squared_error(y_val, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_r2 = r2_score(y_val, lasso_pred)

# Кількість ознак з ненульовими коефіцієнтами
n_features = np.sum(lasso_best.coef_ != 0)

print(f"LASSO регресія (alpha={best_alpha}):")
print(f"RMSE: {lasso_rmse:.6f}")
print(f"R²: {lasso_r2:.6f}")
print(f"Кількість використаних ознак: {n_features} з {X_train_scaled.shape[1]}")
```

**Результат виконання:**
```
LASSO регресія (alpha=0.001):
RMSE: 0.150267
R²: 0.878999
Кількість використаних ознак: 105 з 245
```

**Інтерпретація:** LASSO регресія з оптимальним параметром alpha = 0.001 показала хороші результати (R² = 0.878999, RMSE = 0.150267), незначно кращі за Ridge регресію. Перевага LASSO полягає в автоматичному відборі ознак: з початкових 245 ознак модель використовує лише 105, що спрощує модель і зменшує ризик перенавчання.

### 5. Elastic Net регресія

**Термін:** Elastic Net — це гібридний метод регуляризації, який поєднує регуляризацію L1 та L2 (LASSO та Ridge), дозволяючи вирішувати недоліки обох методів.

```python
# Створення та навчання моделі Elastic Net
alphas = [0.001, 0.01, 0.1, 1]
l1_ratios = [0.1, 0.5, 0.7, 0.9]

best_r2 = -np.inf
best_params = None

for alpha in alphas:
    for l1_ratio in l1_ratios:
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
        en.fit(X_train_scaled, y_train_split)
        en_pred = en.predict(X_val_scaled)
        en_r2 = r2_score(y_val, en_pred)
        
        if en_r2 > best_r2:
            best_r2 = en_r2
            best_params = (alpha, l1_ratio)

# Навчання найкращої моделі
best_alpha, best_l1_ratio = best_params
en_best = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, random_state=42, max_iter=10000)
en_best.fit(X_train_scaled, y_train_split)
en_pred = en_best.predict(X_val_scaled)

# Оцінка моделі
en_mse = mean_squared_error(y_val, en_pred)
en_rmse = np.sqrt(en_mse)
en_r2 = r2_score(y_val, en_pred)

# Кількість ознак з ненульовими коефіцієнтами
n_features = np.sum(en_best.coef_ != 0)

print(f"Elastic Net регресія (alpha={best_alpha}, l1_ratio={best_l1_ratio}):")
print(f"RMSE: {en_rmse:.6f}")
print(f"R²: {en_r2:.6f}")
print(f"Кількість використаних ознак: {n_features} з {X_train_scaled.shape[1]}")
```

**Результат виконання:**
```
Elastic Net регресія (alpha=0.01, l1_ratio=0.7):
RMSE: 0.150400
R²: 0.878784
Кількість використаних ознак: 121 з 245
```

**Інтерпретація:** Elastic Net з оптимальними параметрами (alpha=0.01, l1_ratio=0.7) показала результати, дуже близькі до LASSO регресії (R² = 0.878784, RMSE = 0.150400). Модель використовує 121 ознаку із 245, забезпечуючи хороший баланс між точністю та простотою моделі.

### 6. Поліноміальна регресія

**Термін:** Поліноміальна регресія — це розширення лінійної регресії, яке моделює нелінійні залежності шляхом підвищення ступеня вхідних ознак.

```python
# Створення поліноміальних ознак (степінь 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled[:, :10])  # Використовуємо лише перші 10 ознак для зменшення розмірності
X_val_poly = poly.transform(X_val_scaled[:, :10])

# Створення та навчання моделі лінійної регресії на поліноміальних ознаках
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_split)

# Прогнозування на валідаційному наборі
poly_pred = poly_model.predict(X_val_poly)

# Оцінка моделі
poly_mse = mean_squared_error(y_val, poly_pred)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_val, poly_pred)

print(f"Поліноміальна регресія (степінь 2):")
print(f"RMSE: {poly_rmse:.6f}")
print(f"R²: {poly_r2:.6f}")
```

**Результат виконання:**
```
Поліноміальна регресія (степінь 2):
RMSE: 0.182149
R²: 0.822206
```

**Інтерпретація:** Поліноміальна регресія зі степенем 2 на обмеженому наборі ознак показала не найкращі результати (R² = 0.822206, RMSE = 0.182149). Це може бути пов'язано з використанням лише 10 ознак або з недостатнім підбором гіперпараметрів.

### 7. Стохастичний градієнтний спуск

**Термін:** Стохастичний градієнтний спуск (SGD) — це оптимізаційний алгоритм, який використовується для мінімізації функції втрат. На відміну від звичайного градієнтного спуску, SGD оновлює параметри моделі на основі окремих прикладів або невеликих батчів, а не всього набору даних.

```python
# Створення та навчання моделі з використанням SGD
sgd = SGDRegressor(
    loss='squared_error', 
    penalty='elasticnet',
    alpha=0.01,
    l1_ratio=0.7,  # Використовуємо Elastic Net як регуляризацію
    max_iter=1000,
    tol=1e-3,
    random_state=42
)
sgd.fit(X_train_scaled, y_train_split)

# Прогнозування на валідаційному наборі
sgd_pred = sgd.predict(X_val_scaled)

# Оцінка моделі
sgd_mse = mean_squared_error(y_val, sgd_pred)
sgd_rmse = np.sqrt(sgd_mse)
sgd_r2 = r2_score(y_val, sgd_pred)

print(f"Стохастичний градієнтний спуск:")
print(f"RMSE: {sgd_rmse:.6f}")
print(f"R²: {sgd_r2:.6f}")
```

**Результат виконання:**
```
Стохастичний градієнтний спуск:
RMSE: 17.443303
R²: -1629.497954
```

**Інтерпретація:** Модель на основі стохастичного градієнтного спуску показала дуже погані результати з великим негативним значенням R² (-1629.497954) та високим RMSE (17.443303). Це вказує на нестабільність моделі і на те, що оптимізаційний процес не зійшовся належним чином. Можливо, потрібне краще налаштування гіперпараметрів або нормалізація даних.

### 8. Штучні нейронні мережі

**Термін:** Штучні нейронні мережі (ШНМ) — це обчислювальні системи, натхненні біологічними нейронними мережами. Вони складаються з взаємопов'язаних вузлів (нейронів), організованих у шари, і здатні моделювати складні нелінійні залежності.

```python
# Створення та навчання нейронної мережі
nn_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Два прихованих шари: 100 та 50 нейронів
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    max_iter=500,
    random_state=42,
    early_stopping=True
)
nn_model.fit(X_train_scaled, y_train_split)

# Прогнозування на валідаційному наборі
nn_pred = nn_model.predict(X_val_scaled)

# Оцінка моделі
nn_mse = mean_squared_error(y_val, nn_pred)
nn_rmse = np.sqrt(nn_mse)
nn_r2 = r2_score(y_val, nn_pred)

print(f"Штучна нейронна мережа:")
print(f"RMSE: {nn_rmse:.6f}")
print(f"R²: {nn_r2:.6f}")
```

**Результат виконання:**
```
Штучна нейронна мережа:
RMSE: 0.852300
R²: -2.892676
```

**Інтерпретація:** Штучна нейронна мережа показала незадовільні результати з негативним R² (-2.892676) і дуже високим RMSE (0.852300). Це може свідчити про перенавчання мережі, невідповідність архітектури або недостатню кількість тренувальних даних. Для покращення результатів потрібно краще налаштувати гіперпараметри мережі.

Я розумію, що вам потрібно продовжити переробку файлу з 9-го пункту "Random Forest регресор". Базуючись на інформації з документу і наданих статистичних результатах, підготую оновлену версію цього розділу.

### 9. Random Forest регресор

**Термін:** Random Forest (випадковий ліс) — це ансамблевий метод машинного навчання, який будує множину дерев рішень під час навчання і видає середнє значення прогнозів окремих дерев для регресійних задач.

```python
# Створення та навчання моделі Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,  # Кількість дерев у лісі
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1  # Використання всіх доступних ядер процесора
)
rf_model.fit(X_train_scaled, y_train_split)

# Прогнозування на валідаційному наборі
rf_pred = rf_model.predict(X_val_scaled)

# Оцінка моделі
rf_mse = mean_squared_error(y_val, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_val, rf_pred)

print(f"Random Forest регресор:")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R²: {rf_r2:.4f}")

# Визначення важливості ознак
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
top_features = feature_importances.nlargest(15)

# Візуалізація важливості ознак
plt.figure(figsize=(12, 8))
top_features.plot(kind='barh')
plt.title('15 найважливіших ознак у моделі Random Forest')
plt.xlabel('Важливість')
plt.ylabel('Ознака')
plt.tight_layout()
plt.show()
```

**Результат виконання:**
```
Random Forest регресор:
RMSE: 0.1473
R²: 0.8837
```

**[МІСЦЕ ДЛЯ ГРАФІКА: важливість ознак у моделі Random Forest]**

**Інтерпретація:** Random Forest показав дуже хороші результати (R² = 0.8837, RMSE = 0.1473), що свідчить про ефективність ансамблевих методів для даної задачі. Хоча в порівнянні з робастною регресією RANSAC значення R² трохи нижче (0.8837 проти 0.8995), модель все одно показує високу точність. Додаткова перевага Random Forest — можливість аналізу важливості ознак, що допомагає краще зрозуміти, які фактори найбільше впливають на ціни будинків.

### 10. Метод опорних векторів

**Термін:** Метод опорних векторів (Support Vector Machine, SVM) — це алгоритм машинного навчання, який знаходить гіперплощину в просторі ознак для розділення класів (для класифікації) або для регресії. У випадку регресії алгоритм намагається знайти гіперплощину, яка найкраще відповідає даним з допустимою помилкою.

```python
# Через велику кількість ознак і складність обчислень, виконаємо SVM на підмножині даних
# Використаємо найважливіші ознаки з Random Forest
top_feature_names = top_features.index.tolist()
X_train_svm = X_train_scaled[:, np.array([X_train.columns.get_loc(feat) for feat in top_feature_names])]
X_val_svm = X_val_scaled[:, np.array([X_train.columns.get_loc(feat) for feat in top_feature_names])]

# Створення та навчання моделі SVM
svm_model = SVR(
    kernel='rbf',  # Радіальна базисна функція (RBF)
    C=10,          # Параметр регуляризації
    gamma='scale', # Коефіцієнт ядра RBF
    epsilon=0.1    # Допустиме відхилення
)
svm_model.fit(X_train_svm, y_train_split)

# Прогнозування на валідаційному наборі
svm_pred = svm_model.predict(X_val_svm)

# Оцінка моделі
svm_mse = mean_squared_error(y_val, svm_pred)
svm_rmse = np.sqrt(svm_mse)
svm_r2 = r2_score(y_val, svm_pred)

print(f"Метод опорних векторів (SVR):")
print(f"RMSE: {svm_rmse:.4f}")
print(f"R²: {svm_r2:.4f}")
```

**Результат виконання:**
```
Метод опорних векторів (SVR):
RMSE: 0.1558
R²: 0.8699
```

**Інтерпретація:** Метод опорних векторів показав досить хороші результати (R² = 0.8699, RMSE = 0.1558), хоча й не перевершив результати Random Forest і робастної регресії RANSAC. SVR є обчислювально складним алгоритмом для великих наборів даних, тому для повного набору ознак використовувались лише найважливіші ознаки, визначені за допомогою Random Forest. Незважаючи на це обмеження, модель показала хорошу точність прогнозування.

## Порівняння результатів моделей

Зведемо результати всіх реалізованих моделей для порівняння:

```python
# Створення DataFrame для порівняння результатів
models = ['Лінійна регресія', 'Робастна регресія (RANSAC)', 'Ridge регресія', 
          'LASSO регресія', 'Elastic Net регресія', 'Поліноміальна регресія',
          'Стохастичний градієнтний спуск', 'Штучна нейронна мережа',
          'Random Forest регресор', 'Метод опорних векторів (SVR)']

rmse_scores = [lr_rmse, ransac_rmse, ridge_rmse, lasso_rmse, en_rmse, poly_rmse,
              sgd_rmse, nn_rmse, rf_rmse, svm_rmse]

r2_scores = [lr_r2, ransac_r2, ridge_r2, lasso_r2, en_r2, poly_r2,
            sgd_r2, nn_r2, rf_r2, svm_r2]

results_df = pd.DataFrame({
    'Модель': models,
    'RMSE': rmse_scores,
    'R²': r2_scores
})

# Сортування за R²
results_df = results_df.sort_values(by='R²', ascending=False)

print("Порівняння моделей регресії:")
print(results_df)

# Візуалізація результатів
plt.figure(figsize=(14, 6))

# Графік для R²
plt.subplot(1, 2, 1)
plt.barh(results_df['Модель'], results_df['R²'])
plt.xlabel('R² (вище - краще)')
plt.title('Порівняння моделей за R²')
plt.grid(True, axis='x')
plt.xlim(0.7, 0.95)  # Встановлення діапазону для кращої візуалізації (позитивні значення R²)

# Графік для RMSE
plt.subplot(1, 2, 2)
plt.barh(results_df['Модель'], results_df['RMSE'])
plt.xlabel('RMSE (нижче - краще)')
plt.title('Порівняння моделей за RMSE')
plt.grid(True, axis='x')
plt.xlim(0, 0.3)  # Встановлення діапазону для кращої візуалізації (виключаючи екстремальні значення)

plt.tight_layout()
plt.show()
```

**Результат виконання:**
```
Порівняння моделей регресії:
                          Модель      RMSE         R²
1      Робастна регресія (RANSAC)  0.136958   0.899484
8        Random Forest регресор    0.147302   0.883727
3              LASSO регресія      0.150267   0.878999
4          Elastic Net регресія    0.150400   0.878784
2               Ridge регресія     0.152179   0.875900
9    Метод опорних векторів (SVR)  0.155812   0.869904
5        Поліноміальна регресія    0.182149   0.822206
0              Лінійна регресія    0.215421   0.751322
7       Штучна нейронна мережа     0.852300  -2.892676
6  Стохастичний градієнтний спуск 17.443303 -1629.497954
```

**[МІСЦЕ ДЛЯ ГРАФІКІВ: порівняння моделей за R² та RMSE]**

## Висновки

У цьому блокноті ми розглянули 10 різних алгоритмів регресійного аналізу для прогнозування цін на будинки:

1. **Робастна регресія (RANSAC)** показала найкращі результати (R² = 0.899484, RMSE = 0.136958), що свідчить про наявність викидів у даних, з якими цей метод ефективно справляється.

2. **Random Forest** посів друге місце за ефективністю (R² = 0.883727, RMSE = 0.147302), демонструючи високу точність ансамблевих методів для даного завдання.

3. **LASSO регресія** та **Elastic Net регресія** показали майже однакові результати (R² близько 0.879), поєднуючи хорошу передбачувальну силу з відбором ознак, що робить моделі більш інтерпретованими.

4. **Ridge регресія** також показала високі результати (R² = 0.875900), трохи поступаючись LASSO та Elastic Net.

5. **Метод опорних векторів (SVR)** показав хороші результати (R² = 0.869904) навіть при використанні обмеженого набору ознак.

6. **Поліноміальна регресія** та **Лінійна регресія** показали прийнятні, але менш точні результати (R² = 0.822206 та R² = 0.751322 відповідно).

7. **Штучна нейронна мережа** та **Стохастичний градієнтний спуск** показали незадовільні результати з негативними значеннями R², що свідчить про значні проблеми з налаштуванням або стабільністю цих моделей для даного завдання.

8. Використання логарифмічного перетворення цільової змінної (ціни будинків) значно покращило якість моделей, наблизивши розподіл до нормального.

9. Аналіз важливості ознак за допомогою Random Forest дозволив визначити найбільш впливові фактори для цін на будинки, що є корисним для інтерпретації результатів та потенційного спрощення моделей.

Для прогнозування цін на будинки на основі цього датасету найкраще підходять робастна регресія RANSAC та Random Forest, які показали найвищу точність. Також варто відзначити LASSO та Elastic Net регресії, які забезпечують хороший баланс між точністю та інтерпретованістю завдяки автоматичному відбору ознак. Для подальшого покращення результатів можна рекомендувати ансамблі моделей, більш детальне налаштування гіперпараметрів та додаткову інженерію ознак.
