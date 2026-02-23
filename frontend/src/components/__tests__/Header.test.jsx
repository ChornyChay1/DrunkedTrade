import React from 'react';
import { render, screen } from '@testing-library/react';
import Header from '../Header';


describe('Header Component', () => {
    test('рендерит заголовок', () => {
        render(<Header currentPrice={null} />);
        expect(screen.getByText('CryptoExplorer')).toBeInTheDocument();
    });

    test('отображает цену когда она есть', () => {
        render(<Header currentPrice={50000.45} />);
        expect(screen.getByText('$50000.45')).toBeInTheDocument();
    });

    test('отображает прочерк когда цены нет', () => {
        render(<Header currentPrice={null} />);
        expect(screen.getByText('$—')).toBeInTheDocument();
    });

    test('отображает BTC/USD текст', () => {
        render(<Header currentPrice={null} />);
        expect(screen.getByText('BTC/USD')).toBeInTheDocument();
    });
});