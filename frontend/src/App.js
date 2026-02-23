// src/App.js
import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import Chart from './components/Chart';
import IndicatorPanel from './components/IndicatorPanel';
import './css/App.css';

function App() {
  const [candles, setCandles] = useState([]);
  const [indicators, setIndicators] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/data');
      const data = await response.json();

      const candlesWithTime = data.candles.map((c, i) => ({
        ...c,
        time: Math.floor(Date.now() / 1000) - (data.candles.length - i) * 60
      }));

      setCandles(candlesWithTime);
      setIndicators(data.indicators);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddIndicator = async (indicator) => {
    try {
      const response = await fetch('http://localhost:8000/indicator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(indicator)
      });

      if (response.ok) {
        fetchData();
      }
    } catch (error) {
      console.error('Error creating indicator:', error);
    }
  };

  const handleDeleteIndicator = async (id) => {
    try {
      const response = await fetch(`http://localhost:8000/indicator/${id}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        fetchData();
      }
    } catch (error) {
      console.error('Error deleting indicator:', error);
    }
  };

  const handleUpdateColor = async (id, color) => {
    try {
      const indicator = indicators.find(ind => ind.id === id);
      if (!indicator) return;

      const response = await fetch(`http://localhost:8000/indicator/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: indicator.name,
          type: indicator.type,
          period: indicator.period,
          color: color
        })
      });

      if (response.ok) {
        setIndicators(prev =>
            prev.map(ind =>
                ind.id === id ? { ...ind, color } : ind
            )
        );
      }
    } catch (error) {
      console.error('Error updating color:', error);
    }
  };

  const handleUpdateIndicator = async (id, updatedData) => {
    try {
      const response = await fetch(`http://localhost:8000/indicator/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedData)
      });

      if (response.ok) {
        setIndicators(prev =>
            prev.map(ind =>
                ind.id === id ? { ...ind, ...updatedData } : ind
            )
        );
      }
      fetchData()
    } catch (error) {
      console.error('Error updating indicator:', error);
    }
  };

  const getCurrentPrice = () => {
    if (candles.length === 0) return null;
    return candles[candles.length - 1].close;
  };

  return (
      <div className="app">
        <Header currentPrice={getCurrentPrice()} />

        <main className="main">
          <div className="chart-container">
            {loading && <div className="loading">Загрузка...</div>}
            <Chart candles={candles} indicators={indicators} />
          </div>

          <IndicatorPanel
              indicators={indicators}
              onAdd={handleAddIndicator}
              onDelete={handleDeleteIndicator}
              onUpdateColor={handleUpdateColor}
              onUpdateIndicator={handleUpdateIndicator}
          />
        </main>

        <Footer />
      </div>
  );
}

export default App;