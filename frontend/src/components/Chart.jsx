// src/components/Chart.jsx
import React, { useEffect, useRef } from 'react';
import { createChart, CandlestickSeries, LineSeries } from 'lightweight-charts';

function Chart({ candles, indicators }) {
    const mainChartRef = useRef();
    const mainChartInstance = useRef();
    const subChartInstance = useRef();
    const mainSeriesRef = useRef();
    const indicatorSeriesRef = useRef({});
    const subIndicatorSeriesRef = useRef({});

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

    // Обновление свечей с сохранением масштаба
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
            <div ref={mainChartRef} className="main-chart" />
        </div>
    );
}

export default Chart;