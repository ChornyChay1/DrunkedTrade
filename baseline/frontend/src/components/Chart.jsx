// src/components/Chart.jsx
import React, { useEffect, useRef, useState } from 'react';
import { createChart, CandlestickSeries, LineSeries, createSeriesMarkers } from 'lightweight-charts';
import '../css/App.css';

function Chart({ candles, indicators }) {
    const mainChartRef = useRef();
    const mainChartInstance = useRef();
    const subChartInstance = useRef();
    const mainSeriesRef = useRef();
    const markersApiRef = useRef(); // API для управления маркерами
    const indicatorSeriesRef = useRef({});
    const subIndicatorSeriesRef = useRef({});

    // Состояние для легенды с действиями
    const [actionLegend, setActionLegend] = useState([]);

    // Сохраняем состояние масштаба
    const timeScaleStateRef = useRef(null);

    // Инициализация основного графика
    useEffect(() => {
        if (!mainChartRef.current) return;

        const mainChart = createChart(mainChartRef.current, {
            layout: {
                background: { color: '#ffffff' },
                textColor: '#000000',
            },
            grid: {
                vertLines: { color: '#f0f0f0' },
                horzLines: { color: '#f0f0f0' },
            },
            width: mainChartRef.current.clientWidth,
            height: 500,
            timeScale: {
                borderColor: '#e0e0e0',
                visible: true,
                timeVisible: true,
                secondsVisible: false,
            },
            crosshair: {
                mode: 0,
            },
        });

        mainChartInstance.current = mainChart;

        mainSeriesRef.current = mainChart.addSeries(CandlestickSeries, {
            upColor: '#00c853',
            downColor: '#ff3d00',
            borderDownColor: '#ff3d00',
            borderUpColor: '#00c853',
            wickDownColor: '#000000',
            wickUpColor: '#000000',
        });

        // Сохраняем состояние масштаба при изменении
        const timeScale = mainChart.timeScale();

        const subscribeToVisibleTimeRangeChange = () => {
            timeScale.subscribeVisibleLogicalRangeChange(() => {
                const logicalRange = timeScale.getVisibleLogicalRange();
                if (logicalRange) {
                    timeScaleStateRef.current = {
                        logicalRange,
                        scrollPosition: timeScale.scrollPosition()
                    };
                    // Сохраняем в localStorage
                    try {
                        localStorage.setItem('chartTimeScale', JSON.stringify({
                            logicalRange,
                            timestamp: Date.now()
                        }));
                    } catch (e) {
                        console.error('Error saving to localStorage:', e);
                    }
                }
            });
        };

        subscribeToVisibleTimeRangeChange();

        const handleResize = () => {
            if (mainChartRef.current && mainChartInstance.current) {
                mainChartInstance.current.applyOptions({
                    width: mainChartRef.current.clientWidth,
                });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (mainChartInstance.current) {
                mainChartInstance.current.remove();
            }
        };
    }, []);

    // Функция для форматирования даты
    const formatDate = (timestamp) => {
        const date = new Date(timestamp);
        return date.toLocaleDateString('ru-RU', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    // Функция для создания маркеров на основе action (только стрелочки)
    const createActionMarkers = (candles) => {
        return candles
            .filter(c => c.action === 1 || c.action === -1) // Оставляем только свечи с действиями
            .map(c => {
                const time = Math.floor(c.timestamp / 1000);

                if (c.action === 1) {
                    // Зеленая стрелка вверх - покупка (только стрелка, без текста)
                    return {
                        time: time,
                        position: 'belowBar', // Стрелка под свечой для покупки
                        color: '#00c853', // Зеленый цвет
                        shape: 'arrowUp',
                        text: '', // Убираем текст, оставляем только стрелку
                    };
                } else if (c.action === -1) {
                    // Красная стрелка вниз - продажа (только стрелка, без текста)
                    return {
                        time: time,
                        position: 'aboveBar', // Стрелка над свечой для продажи
                        color: '#ff3d00', // Красный цвет
                        shape: 'arrowDown',
                        text: '', // Убираем текст, оставляем только стрелку
                    };
                }
                return null;
            })
            .filter(marker => marker !== null);
    };

    // Обновление легенды с действиями
    useEffect(() => {
        if (!candles.length) return;

        // Создаем легенду для отображения действий
        const actions = candles
            .filter(c => c.action === 1 || c.action === -1)
            .slice(-10) // Показываем последние 10 действий
            .reverse() // Сначала новые
            .map(c => ({
                time: formatDate(c.timestamp),
                type: c.action === 1 ? 'BUY' : 'SELL',
                price: c.close,
                color: c.action === 1 ? '#00c853' : '#ff3d00'
            }));

        setActionLegend(actions);
    }, [candles]);

    // Инициализация маркеров при первой загрузке или при создании серии
    useEffect(() => {
        if (!mainSeriesRef.current) return;

        // Создаем API для маркеров с опциями
        markersApiRef.current = createSeriesMarkers(
            mainSeriesRef.current,
            [], // Начинаем с пустого массива
            {
                autoScale: true, // Маркеры учитываются при автоскейлинге
                zOrder: 'normal' // Нормальный порядок отрисовки
            }
        );

        return () => {
            // Очищаем маркеры при размонтировании
            if (markersApiRef.current) {
                markersApiRef.current.setMarkers([]);
            }
        };
    }, []); // Зависимость пустая, так как mainSeriesRef.current стабилен

    // Обновление свечей с сохранением масштаба и добавлением маркеров
    useEffect(() => {
        if (!candles.length || !mainSeriesRef.current || !mainChartInstance.current) return;

        const chartData = candles.map((c) => ({
            time: Math.floor(c.timestamp / 1000),
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
        }));

        // Сохраняем текущий масштаб перед обновлением
        const currentTimeScale = mainChartInstance.current.timeScale();
        const currentLogicalRange = currentTimeScale.getVisibleLogicalRange();

        // Обновляем данные
        mainSeriesRef.current.setData(chartData);

        // Обновляем маркеры, используя API
        const markers = createActionMarkers(candles);

        if (markersApiRef.current) {
            // Используем setMarkers для полной замены маркеров
            markersApiRef.current.setMarkers(markers);
        } else {
            // Если API еще не создан, создаем его
            markersApiRef.current = createSeriesMarkers(
                mainSeriesRef.current,
                markers,
                {
                    autoScale: true,
                    zOrder: 'normal'
                }
            );
        }

        // Восстанавливаем масштаб после обновления данных
        if (currentLogicalRange) {
            // Небольшая задержка для применения новых данных
            setTimeout(() => {
                try {
                    currentTimeScale.setVisibleLogicalRange(currentLogicalRange);
                } catch (e) {
                    console.error('Error restoring time scale:', e);
                    // Если не удалось восстановить, применяем сохраненный из localStorage
                    try {
                        const saved = localStorage.getItem('chartTimeScale');
                        if (saved) {
                            const { logicalRange } = JSON.parse(saved);
                            if (logicalRange) {
                                currentTimeScale.setVisibleLogicalRange(logicalRange);
                            }
                        }
                    } catch (e) {
                        console.error('Error restoring from localStorage:', e);
                    }
                }
            }, 50);
        } else {
            // Если нет сохраненного масштаба, пробуем загрузить из localStorage
            try {
                const saved = localStorage.getItem('chartTimeScale');
                if (saved) {
                    const { logicalRange } = JSON.parse(saved);
                    if (logicalRange) {
                        setTimeout(() => {
                            currentTimeScale.setVisibleLogicalRange(logicalRange);
                        }, 50);
                    }
                }
            } catch (e) {
                console.error('Error loading from localStorage:', e);
            }
        }
    }, [candles]);

    // Обновление индикаторов с сохранением масштаба
    useEffect(() => {
        if (!candles.length || !mainChartInstance.current) return;

        // Сохраняем текущий масштаб перед очисткой индикаторов
        const currentTimeScale = mainChartInstance.current.timeScale();
        const currentLogicalRange = currentTimeScale.getVisibleLogicalRange();

        // Очищаем старые индикаторы
        Object.values(indicatorSeriesRef.current).forEach(series => {
            if (mainChartInstance.current) {
                try {
                    mainChartInstance.current.removeSeries(series);
                } catch (e) {
                    console.error('Error removing series from main chart:', e);
                }
            }
        });

        Object.values(subIndicatorSeriesRef.current).forEach(series => {
            if (subChartInstance.current) {
                try {
                    subChartInstance.current.removeSeries(series);
                } catch (e) {
                    console.error('Error removing series from sub chart:', e);
                }
            }
        });

        indicatorSeriesRef.current = {};
        subIndicatorSeriesRef.current = {};

        const priceIndicators = indicators.filter(ind => ['sma', 'ema', 'wma'].includes(ind.type));

        const prepareIndicatorData = (indicatorId) => {
            const data = [];
            candles.forEach((c) => {
                const value = c.indicators?.[indicatorId];
                if (value !== null && value !== undefined && !isNaN(value)) {
                    data.push({
                        time: Math.floor(c.timestamp / 1000),
                        value: value
                    });
                }
            });
            return data;
        };

        // Добавляем новые индикаторы
        if (mainChartInstance.current) {
            priceIndicators.forEach(ind => {
                try {
                    const indicatorData = prepareIndicatorData(ind.id);
                    if (indicatorData.length > 0) {
                        const lineSeries = mainChartInstance.current.addSeries(LineSeries, {
                            color: ind.color,
                            lineWidth: 2,
                            priceLineVisible: false,
                            lastValueVisible: true,
                            crosshairMarkerVisible: true,
                            crosshairMarkerRadius: 4,
                        });
                        lineSeries.setData(indicatorData);
                        indicatorSeriesRef.current[ind.id] = lineSeries;
                    }
                } catch (error) {
                    console.error(`Error adding indicator ${ind.name}:`, error);
                }
            });
        }

        // Восстанавливаем масштаб после добавления индикаторов
        if (currentLogicalRange) {
            setTimeout(() => {
                try {
                    currentTimeScale.setVisibleLogicalRange(currentLogicalRange);
                } catch (e) {
                    console.error('Error restoring time scale after indicators:', e);
                }
            }, 100);
        }

    }, [candles, indicators]);

    return (
        <div className="chart-wrapper">
            <div className="chart-container">
                <div ref={mainChartRef} className="main-chart" />

                {/* Легенда для отображения действий */}
                {actionLegend.length > 0 && (
                    <div className="action-legend">
                        <h4>Последние действия</h4>
                        <div className="legend-items">
                            {actionLegend.map((action, index) => (
                                <div key={index} className="legend-item">
                                    <span className="legend-time">{action.time}</span>
                                    <span
                                        className="legend-type"
                                        style={{ color: action.color, fontWeight: 'bold' }}
                                    >
                                        {action.type}
                                    </span>
                                    <span className="legend-price">
                                        {action.price.toFixed(2)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Chart;