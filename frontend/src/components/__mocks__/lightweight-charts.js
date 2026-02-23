// src/__mocks__/lightweight-charts.js
const seriesMock = {
    setData: jest.fn()
};

const timeScaleMock = {
    subscribeVisibleLogicalRangeChange: jest.fn(cb => {
        // сразу вызываем callback чтобы покрыть localStorage
        cb();
    }),
    getVisibleLogicalRange: jest.fn(() => ({ from: 0, to: 10 })),
    setVisibleLogicalRange: jest.fn(),
    scrollPosition: jest.fn(() => 5)
};

const chartMock = {
    addSeries: jest.fn(() => seriesMock),
    removeSeries: jest.fn(),
    timeScale: jest.fn(() => timeScaleMock),
    applyOptions: jest.fn(),
    remove: jest.fn()
};

module.exports = {
    createChart: jest.fn(() => chartMock),
    CandlestickSeries: jest.fn(),
    LineSeries: jest.fn(),

    // экспортируем моки для тестов
    __mocks__: {
        chartMock,
        timeScaleMock,
        seriesMock
    }
};