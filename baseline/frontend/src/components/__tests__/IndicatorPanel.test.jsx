import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import IndicatorPanel from '../IndicatorPanel';

describe('IndicatorPanel', () => {
    const mockIndicators = [
        { id: '1', name: 'SMA 14', type: 'sma', period: 14, color: '#ff0000' }
    ];

    const mockHandlers = {
        onAdd: jest.fn(),
        onDelete: jest.fn(),
        onUpdateIndicator: jest.fn()
    };

    beforeEach(() => {
        jest.clearAllMocks();
    });

    test('1. рендерит заголовок', () => {
        render(<IndicatorPanel indicators={[]} {...mockHandlers} />);
        expect(screen.getByText('Индикаторы')).toBeInTheDocument();
    });

    test('2. показывает пустое состояние', () => {
        render(<IndicatorPanel indicators={[]} {...mockHandlers} />);
        expect(screen.getByText('Нет добавленных индикаторов')).toBeInTheDocument();
    });

    test('3. показывает список индикаторов', () => {
        render(<IndicatorPanel indicators={mockIndicators} {...mockHandlers} />);
        expect(screen.getByText('SMA 14')).toBeInTheDocument();
    });

    test('4. открывает форму добавления', () => {
        render(<IndicatorPanel indicators={[]} {...mockHandlers} />);
        fireEvent.click(screen.getByText('Добавить'));
        expect(screen.getByPlaceholderText('например: SMA 14')).toBeInTheDocument();
    });

    test('5. добавляет индикатор', () => {
        render(<IndicatorPanel indicators={[]} {...mockHandlers} />);

        fireEvent.click(screen.getByText('Добавить'));

        const nameInput = screen.getByPlaceholderText('например: SMA 14');
        fireEvent.change(nameInput, { target: { value: 'Test SMA' } });

        fireEvent.click(screen.getByText('Создать'));

        expect(mockHandlers.onAdd).toHaveBeenCalled();
    });

    test('6. удаляет индикатор', () => {
        render(<IndicatorPanel indicators={mockIndicators} {...mockHandlers} />);

        const deleteButton = screen.getByTitle('Удалить');
        fireEvent.click(deleteButton);

        expect(mockHandlers.onDelete).toHaveBeenCalledWith('1');
    });

    test('7. редактирует индикатор', () => {
        render(<IndicatorPanel indicators={mockIndicators} {...mockHandlers} />);

        fireEvent.click(screen.getByTitle('Редактировать'));

        const nameInput = screen.getByDisplayValue('SMA 14');
        fireEvent.change(nameInput, { target: { value: 'New Name' } });

        fireEvent.click(screen.getByText('Сохранить'));

        expect(mockHandlers.onUpdateIndicator).toHaveBeenCalled();
    });
});