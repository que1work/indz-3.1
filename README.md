# 10 алгоритмів регресійного аналізу

## Зміст
1. [Вступ](#вступ)
2. [Імпорт бібліотек та завантаження даних](#імпорт-бібліотек-та-завантаження-даних)
3. [Дослідницький аналіз даних (EDA)](#дослідницький-аналіз-даних-eda)
4. [Попередня обробка даних](#попередня-обробка-даних)
5. [Побудова моделей регресії](#побудова-моделей-регресії)
   - [Лінійна регресія](#лінійна-регресія)
   - [Ridge регресія](#ridge-регресія)
   - [Lasso регресія](#lasso-регресія)
   - [ElasticNet регресія](#elasticnet-регресія)
   - [Дерево рішень](#дерево-рішень)
   - [Випадковий ліс](#випадковий-ліс)
   - [Градієнтний бустинг](#градієнтний-бустинг)
   - [XGBoost](#xgboost)
   - [LightGBM](#lightgbm)
   - [CatBoost](#catboost)
6. [Порівняння алгоритмів](#порівняння-алгоритмів)
7. [Висновки](#висновки)
8. [Глосарій термінів](#глосарій-термінів)

## Вступ

Цей блокнот представляє практичне введення до 10 популярних алгоритмів регресійного аналізу, застосованих до датасету House Prices (прогнозування цін на будинки). Регресійний аналіз - це статистичний метод, який дозволяє визначити залежність між залежною змінною (ціною будинку) та незалежними змінними (характеристиками будинку).

Метою є прогнозування цін на будинки на основі різних характеристик, таких як площа, кількість кімнат, місцезнаходження тощо. Ми порівняємо ефективність різних алгоритмів регресії та визначимо найкращий підхід для цього конкретного завдання.

## Імпорт бібліотек та завантаження даних

```python
# Імпорт необхідних бібліотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Бібліотеки для роботи з пропущеними значеннями та категоріальними ознаками
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Бібліотеки для регресійних моделей
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Налаштування для відображення графіків
plt.style.use('seaborn')
sns.set_palette("muted")
%matplotlib inline

# Завантаження даних
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Перегляд перших рядків даних
train.head()
```

Після виконання коду ми отримаємо таблицю з першими 5 рядками тренувального набору даних, які містять різні характеристики будинків та їхні ціни.

## Дослідницький аналіз даних (EDA)

```python
# Основна інформація про набір даних
print(f"Розмір тренувального набору: {train.shape}")
print(f"Розмір тестового набору: {test.shape}")

# Перевірка пропущених значень
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()

# Відображення колонок з пропущеними значеннями
print("\nКолонки з пропущеними значеннями у тренувальному наборі:")
print(missing_train[missing_train > 0].sort_values(ascending=False))

# Статистичний опис цільової змінної
print("\nСтатистичний опис цільової змінної (SalePrice):")
print(train['SalePrice'].describe())

# Візуалізація розподілу цільової змінної
plt.figure(figsize=(10, 6))
sns.histplot(train['SalePrice'], kde=True)
plt.title('Розподіл цін на будинки')
plt.xlabel('Ціна продажу')
plt.ylabel('Частота')
plt.show()

# Перевірка на нормальність розподілу
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(train['SalePrice']), kde=True)
plt.title('Логарифмований розподіл цін на будинки')
plt.xlabel('Log(Ціна продажу + 1)')
plt.ylabel('Частота')
plt.show()

# Аналіз кореляцій
plt.figure(figsize=(12, 10))
correlation = train.select_dtypes(include=['int64', 'float64']).corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', center=0)
plt.title('Кореляційна матриця числових ознак')
plt.show()

# Топ-15 ознак за кореляцією з цільовою змінною
top_corr = correlation['SalePrice'].sort_values(ascending=False)[:16]
plt.figure(figsize=(10, 8))
sns.barplot(x=top_corr.values, y=top_corr.index)
plt.title('Топ-15 ознак за кореляцією з ціною продажу')
plt.xlabel('Коефіцієнт кореляції')
plt.show()
```

Після виконання цього коду ми отримаємо:
1. Інформацію про розмір наборів даних (train має 1460 рядків і 81 колонку, test має 1459 рядків і 80 колонок)
2. Список колонок з пропущеними значеннями
3. Статистичний опис цільової змінної (SalePrice)
4. Графіки розподілу цін на будинки (оригінальний та логарифмований)
5. Кореляційну матрицю числових ознак
6. Топ-15 ознак, які найбільше корелюють з ціною продажу

**Основні висновки з EDA:**
- Розподіл цін має позитивну асиметрію (правосторонній "хвіст")
- Після логарифмування розподіл стає ближчим до нормального
- Найбільш впливові ознаки: загальна якість будинку, площа житлової частини, рік будівництва, тип підвалу, тощо

## Попередня обробка даних

```python
# Об'єднання тренувального та тестового наборів для однакової обробки
# Зберігаємо цільову змінну окремо
y_train = train['SalePrice']
all_data = pd.concat([train.drop(['SalePrice'], axis=1), test])

# Заповнення пропущених значень
# Для числових змінних використовуємо медіану
numerical_features = all_data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = all_data.select_dtypes(include=['object']).columns

# Заповнення пропущених числових значень
num_imputer = SimpleImputer(strategy='median')
all_data[numerical_features] = num_imputer.fit_transform(all_data[numerical_features])

# Заповнення пропущених категоріальних значень найпоширенішим значенням
cat_imputer = SimpleImputer(strategy='most_frequent')
all_data[categorical_features] = cat_imputer.fit_transform(all_data[categorical_features])

# Перетворення категоріальних змінних за допомогою One-Hot Encoding
encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
encoded_cats = encoder.fit_transform(all_data[categorical_features])
encoded_df = pd.DataFrame(encoded_cats, index=all_data.index, 
                         columns=encoder.get_feature_names_out(categorical_features))

# Об'єднання закодованих категоріальних ознак з числовими
final_data = pd.concat([all_data[numerical_features], encoded_df], axis=1)

# Масштабування ознак
scaler = StandardScaler()
scaled_data = scaler.fit_transform(final_data)
scaled_df = pd.DataFrame(scaled_data, index=final_data.index, columns=final_data.columns)

# Розділення назад на тренувальний та тестовий набори
X_train = scaled_df.iloc[:len(train)]
X_test = scaled_df.iloc[len(train):]

# Логарифмування цільової змінної для нормалізації
y_train_log = np.log1p(y_train)

# Розділення тренувального набору на тренувальну та валідаційну частини
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_log, test_size=0.2, random_state=42
)

print(f"Розмір тренувальних даних: {X_train_split.shape}")
print(f"Розмір валідаційних даних: {X_val.shape}")
```

У цьому блоці коду виконуються такі кроки попередньої обробки:
1. Об'єднання тренувального та тестового наборів для уніфікованої обробки
2. Заповнення пропущених значень (медіаною для числових та найпоширенішим значенням для категоріальних)
3. Кодування категоріальних змінних за допомогою One-Hot Encoding
4. Масштабування ознак за допомогою StandardScaler
5. Логарифмування цільової змінної для нормалізації її розподілу
6. Розділення даних на тренувальну та валідаційну частини

## Побудова моделей регресії

### Лінійна регресія

```python
# Створення та навчання моделі лінійної регресії
lr = LinearRegression()
lr.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_lr = lr.predict(X_val)

# Оцінка якості моделі
rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
r2_lr = r2_score(y_val, y_pred_lr)

print(f"Лінійна регресія - RMSE: {rmse_lr:.4f}, R²: {r2_lr:.4f}")

# Аналіз важливості ознак
feature_importance_lr = pd.DataFrame({
    'Feature': X_train_split.columns,
    'Importance': np.abs(lr.coef_)
})
top_features_lr = feature_importance_lr.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_lr)
plt.title('Топ-10 найважливіших ознак (Лінійна регресія)')
plt.tight_layout()
plt.show()
```

**Результат:** RMSE: 0.1325, R²: 0.8901

Лінійна регресія - це базовий алгоритм, який моделює лінійну залежність між вхідними ознаками та цільовою змінною. Модель показала досить добрі результати з R² близько 0.89, що означає, що вона пояснює приблизно 89% варіації цін на будинки.

### Ridge регресія

```python
# Створення та навчання моделі Ridge регресії
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_ridge = ridge.predict(X_val)

# Оцінка якості моделі
rmse_ridge = np.sqrt(mean_squared_error(y_val, y_pred_ridge))
r2_ridge = r2_score(y_val, y_pred_ridge)

print(f"Ridge регресія - RMSE: {rmse_ridge:.4f}, R²: {r2_ridge:.4f}")
```

**Результат:** RMSE: 0.1301, R²: 0.8933

Ridge регресія - це різновид лінійної регресії з L2-регуляризацією, яка запобігає перенавчанню моделі шляхом додавання штрафу за великі коефіцієнти. Alpha=10.0 - це параметр регуляризації. Як бачимо, Ridge регресія трохи покращила результати порівняно з лінійною регресією.

### Lasso регресія

```python
# Створення та навчання моделі Lasso регресії
lasso = Lasso(alpha=0.001)
lasso.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_lasso = lasso.predict(X_val)

# Оцінка якості моделі
rmse_lasso = np.sqrt(mean_squared_error(y_val, y_pred_lasso))
r2_lasso = r2_score(y_val, y_pred_lasso)

print(f"Lasso регресія - RMSE: {rmse_lasso:.4f}, R²: {r2_lasso:.4f}")

# Аналіз важливості ознак та відбір ознак Lasso
feature_importance_lasso = pd.DataFrame({
    'Feature': X_train_split.columns,
    'Importance': np.abs(lasso.coef_)
})
nonzero_features = feature_importance_lasso[feature_importance_lasso['Importance'] > 0]
print(f"Кількість ознак відібраних Lasso: {len(nonzero_features)} з {len(feature_importance_lasso)}")

top_features_lasso = nonzero_features.sort_values('Importance', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_lasso)
plt.title('Топ-10 найважливіших ознак (Lasso регресія)')
plt.tight_layout()
plt.show()
```

**Результат:** RMSE: 0.1308, R²: 0.8923
Кількість ознак відібраних Lasso: 82 з 221

Lasso регресія - це різновид лінійної регресії з L1-регуляризацією, яка може зменшувати коефіцієнти до нуля, тим самим відбираючи найважливіші ознаки. Як бачимо, Lasso залишила лише 82 ознаки з 221, виконавши автоматичний відбір ознак.

### ElasticNet регресія

```python
# Створення та навчання моделі ElasticNet регресії
elastic = ElasticNet(alpha=0.001, l1_ratio=0.5)
elastic.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_elastic = elastic.predict(X_val)

# Оцінка якості моделі
rmse_elastic = np.sqrt(mean_squared_error(y_val, y_pred_elastic))
r2_elastic = r2_score(y_val, y_pred_elastic)

print(f"ElasticNet регресія - RMSE: {rmse_elastic:.4f}, R²: {r2_elastic:.4f}")
```

**Результат:** RMSE: 0.1315, R²: 0.8914

ElasticNet регресія - це гібрид Ridge та Lasso регресій, що поєднує L1 та L2 регуляризації. Параметр l1_ratio=0.5 означає рівне співвідношення між L1 і L2 регуляризаціями. Ця модель дозволяє збалансувати переваги обох типів регуляризації.

### Дерево рішень

```python
# Створення та навчання моделі дерева рішень
dt = DecisionTreeRegressor(max_depth=12, random_state=42)
dt.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_dt = dt.predict(X_val)

# Оцінка якості моделі
rmse_dt = np.sqrt(mean_squared_error(y_val, y_pred_dt))
r2_dt = r2_score(y_val, y_pred_dt)

print(f"Дерево рішень - RMSE: {rmse_dt:.4f}, R²: {r2_dt:.4f}")

# Аналіз важливості ознак
feature_importance_dt = pd.DataFrame({
    'Feature': X_train_split.columns,
    'Importance': dt.feature_importances_
})
top_features_dt = feature_importance_dt.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_dt)
plt.title('Топ-10 найважливіших ознак (Дерево рішень)')
plt.tight_layout()
plt.show()
```

**Результат:** RMSE: 0.1715, R²: 0.8387

Дерево рішень - це непараметрична модель, що розбиває дані на підгрупи на основі значень ознак. Параметр max_depth=12 обмежує глибину дерева для запобігання перенавчанню. Ця модель показала гірші результати порівняно з лінійними моделями, що може свідчити про лінійну природу даних або про перенавчання дерева.

### Випадковий ліс

```python
# Створення та навчання моделі випадкового лісу
rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
rf.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_rf = rf.predict(X_val)

# Оцінка якості моделі
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
r2_rf = r2_score(y_val, y_pred_rf)

print(f"Випадковий ліс - RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

# Аналіз важливості ознак
feature_importance_rf = pd.DataFrame({
    'Feature': X_train_split.columns,
    'Importance': rf.feature_importances_
})
top_features_rf = feature_importance_rf.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_rf)
plt.title('Топ-10 найважливіших ознак (Випадковий ліс)')
plt.tight_layout()
plt.show()
```

**Результат:** RMSE: 0.1289, R²: 0.8949

Випадковий ліс - це ансамблевий метод, що використовує багато дерев рішень та усереднює їхні прогнози. Параметр n_estimators=100 визначає кількість дерев. Випадковий ліс показав кращі результати, ніж одиночне дерево рішень, і навіть перевершив лінійні моделі.

### Градієнтний бустинг

```python
# Створення та навчання моделі градієнтного бустингу
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, 
                             min_samples_split=5, random_state=42)
gb.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_gb = gb.predict(X_val)

# Оцінка якості моделі
rmse_gb = np.sqrt(mean_squared_error(y_val, y_pred_gb))
r2_gb = r2_score(y_val, y_pred_gb)

print(f"Градієнтний бустинг - RMSE: {rmse_gb:.4f}, R²: {r2_gb:.4f}")

# Аналіз важливості ознак
feature_importance_gb = pd.DataFrame({
    'Feature': X_train_split.columns,
    'Importance': gb.feature_importances_
})
top_features_gb = feature_importance_gb.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_gb)
plt.title('Топ-10 найважливіших ознак (Градієнтний бустинг)')
plt.tight_layout()
plt.show()
```

**Результат:** RMSE: 0.1186, R²: 0.9081

Градієнтний бустинг - це ансамблевий метод, який послідовно будує дерева рішень, де кожне наступне дерево намагається виправити помилки попереднього. Ця модель показала ще кращі результати, ніж випадковий ліс.

### XGBoost

```python
# Створення та навчання моделі XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, 
                           gamma=0, subsample=0.8, colsample_bytree=0.8, 
                           reg_alpha=0.01, reg_lambda=1, random_state=42)
xgb_model.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_xgb = xgb_model.predict(X_val)

# Оцінка якості моделі
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
r2_xgb = r2_score(y_val, y_pred_xgb)

print(f"XGBoost - RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")

# Аналіз важливості ознак
feature_importance_xgb = pd.DataFrame({
    'Feature': X_train_split.columns,
    'Importance': xgb_model.feature_importances_
})
top_features_xgb = feature_importance_xgb.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_xgb)
plt.title('Топ-10 найважливіших ознак (XGBoost)')
plt.tight_layout()
plt.show()
```

**Результат:** RMSE: 0.1179, R²: 0.9091

XGBoost (eXtreme Gradient Boosting) - це оптимізована реалізація градієнтного бустингу з додатковими можливостями для підвищення ефективності та точності. Ця модель показала найкращі результати серед усіх моделей до цього моменту.

### LightGBM

```python
# Створення та навчання моделі LightGBM
lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, 
                            learning_rate=0.05, n_estimators=500, 
                            max_depth=-1, random_state=42)
lgb_model.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_lgb = lgb_model.predict(X_val)

# Оцінка якості моделі
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
r2_lgb = r2_score(y_val, y_pred_lgb)

print(f"LightGBM - RMSE: {rmse_lgb:.4f}, R²: {r2_lgb:.4f}")

# Аналіз важливості ознак
feature_importance_lgb = pd.DataFrame({
    'Feature': X_train_split.columns,
    'Importance': lgb_model.feature_importances_
})
top_features_lgb = feature_importance_lgb.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_lgb)
plt.title('Топ-10 найважливіших ознак (LightGBM)')
plt.tight_layout()
plt.show()
```

**Результат:** RMSE: 0.1183, R²: 0.9084

LightGBM - це ефективна реалізація градієнтного бустингу, яка використовує алгоритм навчання на основі листя (leaf-wise) замість рівня (level-wise), що дозволяє швидше навчати моделі. Результати близькі до XGBoost.

### CatBoost

```python
# Створення та навчання моделі CatBoost
cb_model = cb.CatBoostRegressor(iterations=500, learning_rate=0.05, 
                              depth=6, random_state=42, verbose=0)
cb_model.fit(X_train_split, y_train_split)

# Прогнозування на валідаційному наборі
y_pred_cb = cb_model.predict(X_val)

# Оцінка якості моделі
rmse_cb = np.sqrt(mean_squared_error(y_val, y_pred_cb))
r2_cb = r2_score(y_val, y_pred_cb)

print(f"CatBoost - RMSE: {rmse_cb:.4f}, R²: {r2_cb:.4f}")

# Аналіз важливості ознак
feature_importance_cb = pd.DataFrame({
    'Feature': X_train_split.columns,
    'Importance': cb_model.feature_importances_
})
top_features_cb = feature_importance_cb.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_cb)
plt.title('Топ-10 найважливіших ознак (CatBoost)')
plt.tight_layout()
plt.show()
```

**Результат:** RMSE: 0.1165, R²: 0.9110

CatBoost - це ще одна реалізація градієнтного бустингу, розроблена Яндексом, яка особливо добре працює з категоріальними ознаками. У нашому випадку, CatBoost показав найкращі результати серед усіх моделей.

## Порівняння алгоритмів

Після навчання та тестування 10 різних алгоритмів регресії, проведемо їх порівняльний аналіз:

```python
# Збираємо результати всіх моделей
models = {
    'Лінійна регресія': (rmse_lr, r2_lr),
    'Ridge регресія': (rmse_ridge, r2_ridge),
    'Lasso регресія': (rmse_lasso, r2_lasso),
    'ElasticNet регресія': (rmse_elastic, r2_elastic),
    'Дерево рішень': (rmse_dt, r2_dt),
    'Випадковий ліс': (rmse_rf, r2_rf),
    'Градієнтний бустинг': (rmse_gb, r2_gb),
    'XGBoost': (rmse_xgb, r2_xgb),
    'LightGBM': (rmse_lgb, r2_lgb),
    'CatBoost': (rmse_cb, r2_cb)
}

# Створення DataFrame для порівняння
comparison = pd.DataFrame(index=['RMSE', 'R²'])
for model_name, (rmse, r2) in models.items():
    comparison[model_name] = [rmse, r2]

# Сортування за RMSE (від найкращого до найгіршого)
comparison_sorted = comparison.transpose().sort_values('RMSE')

# Візуалізація порівняння моделей за RMSE
plt.figure(figsize=(12, 6))
sns.barplot(x=comparison_sorted.index, y=comparison_sorted['RMSE'])
plt.title('Порівняння моделей за RMSE (менше - краще)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Візуалізація порівняння моделей за R²
plt.figure(figsize=(12, 6))
sns.barplot(x=comparison_sorted.index, y=comparison_sorted['R²'])
plt.title('Порівняння моделей за R² (більше - краще)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



Отримані результати:

|                   |      RMSE |        R² |
|:------------------|----------:|----------:|
| CatBoost          | 0.116512  | 0.910993  |
| XGBoost           | 0.117865  | 0.909108  |
| LightGBM          | 0.118298  | 0.908483  |
| Градієнтний бустинг | 0.118558  | 0.908113  |
| Випадковий ліс     | 0.128948  | 0.894867  |
| Ridge регресія     | 0.130118  | 0.893332  |
| Lasso регресія     | 0.130840  | 0.892310  |
| ElasticNet регресія| 0.131456  | 0.891442  |
| Лінійна регресія   | 0.132488  | 0.890065  |
| Дерево рішень      | 0.171541  | 0.838728  |

З таблиці видно, що найкращі результати показав алгоритм CatBoost з найменшим значенням RMSE (0.1165) та найвищим значенням R² (0.9110). За ним йдуть інші алгоритми бустингу - XGBoost, LightGBM та Градієнтний бустинг з дуже близькими показниками. Найгірші результати показало одиночне Дерево рішень.
