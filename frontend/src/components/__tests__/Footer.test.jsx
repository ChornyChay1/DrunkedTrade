import React from 'react';
import { render, screen } from '@testing-library/react';
import Footer from '../Footer';

describe('Footer Component', () => {
    test('renders copyright text', () => {
        render(<Footer />);
        expect(screen.getByText(/CryptoExplorer © 2026/)).toBeInTheDocument();
    });

    test('renders update interval text', () => {
        render(<Footer />);
        expect(screen.getByText(/Данные обновляются каждые 5 секунд/)).toBeInTheDocument();
    });

    test('has footer class', () => {
        render(<Footer />);
        expect(screen.getByRole('contentinfo')).toHaveClass('footer');
    });
});