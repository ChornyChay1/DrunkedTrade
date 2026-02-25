import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # Для балансировки классов
import joblib
import os
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger("ML-Service")


class TradingTreeModel:
    def __init__(self, model_path: str = "models/tree_model.joblib", scaler_path: str = "models/scaler.joblib"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        # Используем RandomForest с балансировкой весов
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',  # Балансировка классов
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_samples = 0
        self.feature_names = [
            'returns_1', 'returns_3', 'returns_5',
            'volume_change', 'volume_ratio',
            'high_low_ratio', 'close_position',
            'volatility', 'rsi', 'macd',
            'sma_5_ratio', 'sma_10_ratio'
        ]

        os.makedirs("models", exist_ok=True)

    def _prepare_features(self, candles: List[dict]) -> np.ndarray:
        """
        Подготовка признаков
        """
        df = pd.DataFrame(candles)

        # Ценовые признаки
        df['returns_1'] = df['close'].pct_change().fillna(0)
        df['returns_3'] = df['close'].pct_change(3).fillna(0)
        df['returns_5'] = df['close'].pct_change(5).fillna(0)

        # Объем
        df['volume_ma'] = df['volume'].rolling(5).mean().fillna(df['volume'].mean())
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        df['volume_ratio'] = (df['volume'] / df['volume_ma']).fillna(1)

        # Волатильность
        df['high_low'] = df['high'] - df['low']
        df['high_low_ratio'] = (df['high_low'] / df['close']).fillna(0)
        df['close_position'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)).fillna(0.5)
        df['volatility'] = df['returns_1'].rolling(5).std().fillna(0)

        # Простые скользящие средние
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_5_ratio'] = ((df['close'] - df['sma_5']) / df['sma_5']).fillna(0)
        df['sma_10_ratio'] = ((df['close'] - df['sma_10']) / df['sma_10']).fillna(0)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (exp1 - exp2).fillna(0)

        features = df[self.feature_names].values

        return features

    def _generate_signals(self, prices: np.ndarray, volumes: np.ndarray, lookahead: int = 3) -> np.ndarray:
        """
        Улучшенная генерация сигналов с учетом волатильности
        """
        signals = np.zeros(len(prices))

        # Динамические пороги на основе волатильности
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Годовая волатильность

        # Адаптивные пороги
        buy_threshold = max(0.005, volatility * 0.5)  # Не меньше 0.5%
        sell_threshold = -max(0.005, volatility * 0.5)

        for i in range(20, len(prices) - lookahead):
            # Будущая доходность
            future_return = (prices[i + lookahead] - prices[i]) / prices[i]

            # Проверяем объем для подтверждения сигнала
            volume_confirm = volumes[i] > np.mean(volumes[i - 5:i])

            # Генерируем сигналы
            if future_return > buy_threshold and volume_confirm:
                signals[i] = 1  # Покупка
            elif future_return < sell_threshold and volume_confirm:
                signals[i] = -1  # Продажа
            else:
                # Проверяем на сильные движения без подтверждения объемом
                if future_return > buy_threshold * 2:
                    signals[i] = 1  # Сильное движение без объема
                elif future_return < sell_threshold * 2:
                    signals[i] = -1
                else:
                    signals[i] = 0  # Держать

        return signals

    def train(self, candles: List[dict]) -> Tuple[float, int]:
        """
        Обучение модели с балансировкой классов
        """
        if len(candles) < 50:
            raise ValueError("Need at least 50 candles for training")

        # Подготовка признаков
        features = self._prepare_features(candles)

        # Генерация сигналов
        prices = np.array([c['close'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])
        targets = self._generate_signals(prices, volumes)

        # Обрезаем начало и конец
        min_len = min(len(features), len(targets))
        start_idx = 30
        end_idx = min_len - 5
        features = features[start_idx:end_idx]
        targets = targets[start_idx:end_idx]

        # Проверяем распределение классов
        unique, counts = np.unique(targets, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Initial class distribution: {class_dist}")

        # Если есть дисбаланс, применяем SMOTE
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, sampling_strategy='auto')

        try:
            features_balanced, targets_balanced = smote.fit_resample(features, targets)
            logger.info(f"Balanced class distribution: {dict(zip(*np.unique(targets_balanced, return_counts=True)))}")
        except:
            # Если SMOTE не сработал, используем исходные данные
            features_balanced, targets_balanced = features, targets
            logger.warning("SMOTE failed, using original data")

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features_balanced, targets_balanced,
            test_size=0.2,
            random_state=42,
            stratify=targets_balanced
        )

        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучение
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        self.training_samples = len(features_balanced)

        # Оценка
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Детальная метрика по классам
        from sklearn.metrics import classification_report
        y_pred = self.model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(f"Model trained: {len(features_balanced)} samples")
        logger.info(f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        logger.info(f"Classification report: {report}")

        return test_score, len(features_balanced)

    def predict(self, candles: List[dict]) -> int:
        """
        Предсказание с проверкой уверенности
        """
        if not self.is_trained or len(candles) < 20:
            return self._momentum_strategy(candles)

        # Подготовка признаков
        features = self._prepare_features(candles)
        last_features = features[-1:].reshape(1, -1)
        last_features_scaled = self.scaler.transform(last_features)

        # Предсказание с вероятностями
        prediction = self.model.predict(last_features_scaled)[0]
        probabilities = self.model.predict_proba(last_features_scaled)[0]

        # Уверенность в предсказании
        confidence = max(probabilities)

        # Логируем для отладки
        logger.debug(f"Prediction: {prediction}, Confidence: {confidence:.3f}")
        logger.debug(f"Probabilities: {probabilities}")

        # Возвращаем предсказание только если уверенность > 0.4
        if confidence > 0.4:
            return int(prediction)
        else:
            # Если низкая уверенность, используем моментум стратегию
            momentum_pred = self._momentum_strategy(candles)
            logger.debug(f"Low confidence, using momentum: {momentum_pred}")
            return momentum_pred

    def _momentum_strategy(self, candles: List[dict]) -> int:
        """
        Стратегия на основе моментума и RSI
        """
        if len(candles) < 10:
            return 0

        df = pd.DataFrame(candles[-14:])

        # Рассчитываем RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Рассчитываем моментум
        returns_3 = (df['close'].iloc[-1] / df['close'].iloc[-4] - 1) if len(df) >= 4 else 0
        returns_5 = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) if len(df) >= 6 else 0

        # Скользящие средние
        sma_5 = df['close'].tail(5).mean()
        sma_10 = df['close'].tail(10).mean() if len(df) >= 10 else sma_5

        # Объемный анализ
        volume_trend = df['volume'].iloc[-1] > df['volume'].tail(5).mean()

        # Принимаем решение
        if (returns_3 > 0.01 or returns_5 > 0.015) and rsi < 70 and sma_5 > sma_10:
            return 1  # Покупка
        elif (returns_3 < -0.01 or returns_5 < -0.015) and rsi > 30 and sma_5 < sma_10:
            return -1  # Продажа
        else:
            # Проверяем на дивергенции
            if returns_3 > 0.005 and volume_trend:
                return 1
            elif returns_3 < -0.005 and volume_trend:
                return -1
            else:
                return 0  # Держать

    def _simple_strategy(self, candles: List[dict]) -> int:
        """
        Простая стратегия как запасной вариант
        """
        if len(candles) < 5:
            return 0

        closes = [c['close'] for c in candles[-5:]]

        # Тренд за 5 свечей
        trend = (closes[-1] - closes[0]) / closes[0]

        # Волатильность
        volatility = np.std([(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, 5)])

        # Адаптивные пороги
        threshold = max(0.005, volatility)

        if trend > threshold:
            return 1
        elif trend < -threshold:
            return -1
        else:
            return 0

    def save_model(self):
        """Сохранение модели"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Загрузка модели"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
            logger.info(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError("Model files not found")


# Глобальный экземпляр модели
model = TradingTreeModel()