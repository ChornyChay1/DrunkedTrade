// src/components/AnalyticsPanel.jsx
import React from 'react';
import '../css/App.css'; // стили можно свои

function AnalyticsPanel({ analytics }) {
    if (!analytics) return null;

    const rows = [
        { label: 'Всего покупок', value: analytics.total_buy_signals },
        { label: 'Всего продаж', value: analytics.total_sell_signals },
        { label: 'Всего удержаний', value: analytics.total_hold_signals },
        { label: 'Средняя цена покупки', value: analytics.avg_buy_price?.toFixed(2) || '0.00' },
        { label: 'Средняя цена продажи', value: analytics.avg_sell_price?.toFixed(2) || '0.00' },
        { label: 'Прибыль стратегии', value: analytics.strategy_profit?.toFixed(2) || '0.00' },
        { label: 'Прибыль по парам', value: analytics.pair_profit?.toFixed(2) || '0.00' },
        { label: 'Завершенных сделок', value: analytics.completed_trades || 0 },
        { label: 'Точность предсказаний', value: analytics.prediction_coverage || '0%' },
        { label: 'Всего свечей', value: analytics.total_candles || 0 },
    ];

    return (
        <div className="analytics-panel">
            <div className="analytics-header"> Аналитика </div>
            <div className="analytics-grid">

                {rows.map((row, idx) => (
                    <div key={idx} className="analytics-row">
                        <div className="analytics-label">{row.label}</div>
                        <div className="analytics-value">{row.value}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default AnalyticsPanel;