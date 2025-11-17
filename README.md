AI Stock Agent Web App

This project is a browser-based stock analysis tool that combines intraday market data, technical indicators, and AI-generated insights. The application is built with Flask, YFinance, Pandas, and an optional LLM provider for natural-language analysis.

Features
Real-Time Market Data

Fetches intraday candles from YFinance

Supports multiple intervals (1m, 2m, 5m, 15m, 30m, 60m)

Displays OHLCV data for the most recent session

Technical Indicators

Automatically calculates:

SMA-10

SMA-20

EMA-10

EMA-20

RSI-14

AI Market Summary

Uses an LLM (or fallback logic) to interpret recent price action and indicators.
Capabilities include:

Trend interpretation

Momentum analysis

Support and resistance identification

Short-term sentiment assessment

Web User Interface

The application includes a simple front-end where users can:

Enter a stock symbol

Select interval and lookback period

Generate an analysis with one click
