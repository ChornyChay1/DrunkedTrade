// src/components/AnalyticsPanel.jsx
import React from 'react';
import '../css/App.css'; // стили можно свои

function AnalyticsPanel({ analytics }) {
    if (!analytics) return null;

    const rows = [
        { label: 'Всего покупок', value: analytics.total_buy },
        { label: 'Всего продаж', value: analytics.total_sell },
        { label: 'Средняя цена покупки', value: analytics.avg_buy.toFixed(2) },
        { label: 'Средняя цена продажи', value: analytics.avg_sell.toFixed(2) },
        { label: 'Прибыль', value: analytics.avg_profit.toFixed(2) },
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