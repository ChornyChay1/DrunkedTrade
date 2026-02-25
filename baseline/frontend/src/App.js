// src/App.js
import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import Chart from './components/Chart';
import IndicatorPanel from './components/IndicatorPanel';
import './css/App.css';
import AnalyticsPanel from "./components/AnalyticsPanel";

const API_URL = 'http://localhost:8000/symbol/data';

function App() {
  const [candles, setCandles] = useState([]);
  const [indicators, setIndicators] = useState([]);
  const [loading, setLoading] = useState(false);
  const [analytics, setAnalytics] = useState(null);

  useEffect(() => {
    fetchData();

    const interval = setInterval(fetchData, 180000);
    return () => clearInterval(interval);
  }, [indicators]); // перезапрашиваем если изменились индикаторы


  const fetchData = async () => {
    try {
      setLoading(true);

      let url = API_URL;

      if (indicators.length > 0) {
        const query = encodeURIComponent(JSON.stringify(indicators));
        url += `?indicators=${query}`;
      }

      const response = await fetch(url);
      const data = await response.json();

      // используем start от Bybit
      const candlesWithTime = data.candles.map(c => ({
        ...c,
        time: Math.floor(c.start / 1000)
      }));

      setCandles(candlesWithTime);
      setAnalytics(data.analytics)
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };


  // --- локальное добавление ---
  const handleAddIndicator = (indicator) => {
    const newIndicator = {
      ...indicator,
      id: crypto.randomUUID()
    };

    setIndicators(prev => [...prev, newIndicator]);
  };


  // --- локальное удаление ---
  const handleDeleteIndicator = (id) => {
    setIndicators(prev =>
        prev.filter(ind => ind.id !== id)
    );
  };


  // --- изменение цвета ---
  const handleUpdateColor = (id, color) => {
    setIndicators(prev =>
        prev.map(ind =>
            ind.id === id ? { ...ind, color } : ind
        )
    );
  };


  // --- изменение параметров ---
  const handleUpdateIndicator = (id, updatedData) => {
    setIndicators(prev =>
        prev.map(ind =>
            ind.id === id ? { ...ind, ...updatedData } : ind
        )
    );
  };


  const getCurrentPrice = () => {
    if (candles.length === 0) return null;
    return candles[candles.length - 1].close;
  };


  return (
      <div className="app">

        <Header currentPrice={getCurrentPrice()} />

        <main className="main">

          <div className="chart-view">
            {loading && <div className="loading">Загрузка...</div>}

            <Chart
                candles={candles}
                indicators={indicators}
            />
          </div>

          <div className="sidebar">

            <IndicatorPanel
                indicators={indicators}
                onAdd={handleAddIndicator}
                onDelete={handleDeleteIndicator}
                onUpdateColor={handleUpdateColor}
                onUpdateIndicator={handleUpdateIndicator}
            />
            <AnalyticsPanel analytics={analytics} />
          </div>
        </main>

        <Footer />

      </div>
  );
}

export default App;