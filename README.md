# DevOpsCryptoBrowser

Веб-приложение для просмотра криптовалютных свечных данных (BTC/USD) с поддержкой технических индикаторов.


---

## Возможности

* Отображение свечного графика BTC/USD
* Автообновление данных каждые 5 секунд
* Добавление/редактирование/удаление индикаторов
* Группировка индикаторов (ценовые / осцилляторы)
* Сохранение состояния

---


# Локальный запуск

## Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API будет доступен на:

```
http://localhost:8000
```

---

## Frontend

```bash
cd frontend
npm install
npm start
```

Приложение откроется на:

```
http://localhost:3000
```


---

# Тестирование

## Запуск тестов

```bash
cd frontend
npm test -- --watchAll=false
```

---

# CI (GitHub Actions)

При каждом:

* push
* pull request

CI выполняет:

1. Установку зависимостей
2. Кэширование зависимостей
3. Запуск тестов

---

# Поддерживаемые индикаторы

### Ценовые

* SMA
* EMA
* WMA

### Осцилляторы

* RSI
* MACD
* Stochastic

---

# Технологии

### Frontend

* React
* lightweight-charts
* Jest
* React Testing Library

### Backend

* FastAPI
* Uvicorn

### DevOps

* GitHub Actions

---

# API

## Получить данные

```
GET /data
```

Ответ:

```json
{
  "candles": [...],
  "indicators": [...]
}
```

## Добавить индикатор

```
POST /indicator
```

## Обновить индикатор

```
PUT /indicator/{id}
```

## Удалить индикатор

```
DELETE /indicator/{id}
```

---

# Авторы

**Артём Ковалёв**

**Никита Кулаков**
