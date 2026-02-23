// src/components/IndicatorPanel.jsx
import React, { useState } from 'react';

function IndicatorPanel({ indicators, onAdd, onDelete, onUpdateIndicator }) {
    const [showAddForm, setShowAddForm] = useState(false);
    const [editingId, setEditingId] = useState(null);
    const [editFormData, setEditFormData] = useState(null); // Данные для редактирования
    const [newIndicator, setNewIndicator] = useState({
        name: '',
        type: 'sma',
        period: 14,
        color: '#2196f3'
    });

    const handleSubmit = () => {
        if (newIndicator.name && newIndicator.period > 0) {
            onAdd(newIndicator);
            setNewIndicator({ name: '', type: 'sma', period: 14, color: '#2196f3' });
            setShowAddForm(false);
        }
    };

    // Начать редактирование
    const startEditing = (indicator) => {
        setEditFormData({ ...indicator }); // Копируем данные индикатора
        setEditingId(indicator.id);
        setShowAddForm(false);
    };

    // Отмена редактирования
    const cancelEditing = () => {
        setEditingId(null);
        setEditFormData(null);
    };

    // Сохранить изменения
    const saveEditing = () => {
        if (editFormData && editFormData.name && editFormData.period > 0) {
            onUpdateIndicator(editFormData.id, {
                name: editFormData.name,
                type: editFormData.type,
                period: editFormData.period,
                color: editFormData.color
            });
            setEditingId(null);
            setEditFormData(null);
        }
    };

    const getDisplayType = (type) => {
        const types = {
            'sma': 'SMA',
            'ema': 'EMA',
            'wma': 'WMA',
            'rsi': 'RSI',
            'roc': 'ROC',
            'momentum': 'MOMENTUM',
            'williams_r': 'WILLIAMS %R',
            'atr': 'ATR',
            'cci': 'CCI'
        };
        return types[type] || type.toUpperCase();
    };

    const getIndicatorCategory = (type) => {
        const priceIndicators = ['sma', 'ema', 'wma'];
        return priceIndicators.includes(type) ? 'price' : 'oscillator';
    };

    const getCategoryName = (category) => {
        return category === 'price' ? 'Ценовые' : 'Осцилляторы';
    };

    const groupedIndicators = indicators.reduce((acc, ind) => {
        const category = getIndicatorCategory(ind.type);
        if (!acc[category]) acc[category] = [];
        acc[category].push(ind);
        return acc;
    }, {});

    return (
        <div className="indicators-panel">
            <div className="panel-header">
                <h3>Индикаторы</h3>
                <button
                    className="add-button"
                    onClick={() => {
                        setShowAddForm(!showAddForm);
                        setEditingId(null);
                        setEditFormData(null);
                    }}
                >
                    <span className="add-button-icon">{showAddForm ? '−' : '+'}</span>
                    <span className="add-button-text">
                        {showAddForm ? 'Закрыть' : 'Добавить'}
                    </span>
                </button>
            </div>

            {showAddForm && (
                <div className="add-form">
                    <div className="add-form-header">
                        <span className="add-form-title">Новый индикатор</span>
                    </div>

                    <div className="form-group">
                        <label>Название</label>
                        <input
                            type="text"
                            placeholder="например: SMA 14"
                            value={newIndicator.name}
                            onChange={(e) => setNewIndicator({...newIndicator, name: e.target.value})}
                        />
                    </div>

                    <div className="form-group">
                        <label>Тип индикатора</label>
                        <select
                            value={newIndicator.type}
                            onChange={(e) => setNewIndicator({...newIndicator, type: e.target.value})}
                        >
                            <optgroup label="Ценовые индикаторы">
                                <option value="sma">SMA</option>
                                <option value="ema">EMA</option>
                                <option value="wma">WMA</option>
                            </optgroup>
                        </select>
                    </div>

                    <div className="form-row">
                        <div className="form-group period-group">
                            <label>Период</label>
                            <input
                                type="number"
                                min="1"
                                max="200"
                                value={newIndicator.period}
                                onChange={(e) => setNewIndicator({...newIndicator, period: parseInt(e.target.value) || 14})}
                            />
                        </div>

                        <div className="form-group color-group">
                            <label>Цвет</label>
                            <div className="color-picker-trigger">
                                <input
                                    type="color"
                                    value={newIndicator.color}
                                    onChange={(e) => setNewIndicator({...newIndicator, color: e.target.value})}
                                />
                                <span
                                    className="color-preview"
                                    style={{ backgroundColor: newIndicator.color }}
                                />
                            </div>
                        </div>
                    </div>

                    <div className="form-actions">
                        <button className="btn btn-secondary" onClick={() => setShowAddForm(false)}>
                            Отмена
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={handleSubmit}
                            disabled={!newIndicator.name}
                        >
                            Создать
                        </button>
                    </div>
                </div>
            )}

            <div className="indicators-list">
                {indicators.length > 0 ? (
                    Object.entries(groupedIndicators).map(([category, cats]) => (
                        <div key={category} className="indicator-category">
                            <div className="category-header">
                                <h4>{getCategoryName(category)}</h4>
                                <span className="category-count">{cats.length}</span>
                            </div>
                            {cats.map(ind => (
                                <div key={ind.id} className="indicator-item">
                                    {editingId === ind.id && editFormData ? (
                                        // Режим редактирования с вашими стилями
                                        <div className="indicator-edit-mode">
                                            <input
                                                type="text"
                                                className="edit-name-input"
                                                value={editFormData.name}
                                                onChange={(e) => setEditFormData({...editFormData, name: e.target.value})}
                                                placeholder="Название"
                                                autoFocus
                                            />

                                            <select
                                                className="edit-type-select"
                                                value={editFormData.type}
                                                onChange={(e) => setEditFormData({...editFormData, type: e.target.value})}
                                            >
                                                <optgroup label="Ценовые">
                                                    <option value="sma">SMA</option>
                                                    <option value="ema">EMA</option>
                                                    <option value="wma">WMA</option>
                                                </optgroup>
                                            </select>

                                            <div className="edit-panel">
                                                <label>Период:</label>
                                                <input
                                                    type="number"
                                                    min="1"
                                                    max="200"
                                                    value={editFormData.period}
                                                    onChange={(e) => setEditFormData({
                                                        ...editFormData,
                                                        period: parseInt(e.target.value) || 1
                                                    })}
                                                />
                                            </div>

                                            <div className="edit-panel">
                                                <label>Цвет:</label>
                                                <input
                                                    type="color"
                                                    value={editFormData.color || '#2196f3'}
                                                    onChange={(e) => setEditFormData({...editFormData, color: e.target.value})}
                                                />
                                            </div>

                                            <div className="edit-actions">
                                                <button
                                                    className="btn btn-secondary"
                                                    onClick={cancelEditing}
                                                >
                                                    Отмена
                                                </button>
                                                <button
                                                    className="btn btn-primary"
                                                    onClick={saveEditing}
                                                    disabled={!editFormData.name}
                                                >
                                                    Сохранить
                                                </button>
                                            </div>
                                        </div>
                                    ) : (
                                        // Обычный режим отображения
                                        <>
                                            <div className="indicator-info">
                                                <div className="indicator-header">
                                                    <span
                                                        className="color-dot"
                                                        style={{ backgroundColor: ind.color || '#2196f3' }}
                                                    />
                                                    <span className="indicator-name">{ind.name}</span>
                                                </div>
                                                <span className="indicator-details">
                                                    {getDisplayType(ind.type)} · Период: {ind.period}
                                                </span>
                                            </div>

                                            <div className="indicator-actions">
                                                <button
                                                    className="icon-button"
                                                    onClick={() => startEditing(ind)}
                                                    title="Редактировать"
                                                >
                                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                                        <path d="M17 3L21 7L7 21H3V17L17 3Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                                        <path d="M14 6L18 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                                    </svg>
                                                </button>

                                                <button
                                                    className="delete-button"
                                                    onClick={() => onDelete(ind.id)}
                                                    title="Удалить"
                                                >
                                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                                                        <path d="M3 6H5H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                                        <path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                                                    </svg>
                                                </button>
                                            </div>
                                        </>
                                    )}
                                </div>
                            ))}
                        </div>
                    ))
                ) : (
                    <div className="empty-state">
                        <p>Нет добавленных индикаторов</p>
                        <button className="btn btn-primary" onClick={() => setShowAddForm(true)}>
                            Добавить первый индикатор
                        </button>
                    </div>
                )}
            </div>

        </div>
    );
}

export default IndicatorPanel;