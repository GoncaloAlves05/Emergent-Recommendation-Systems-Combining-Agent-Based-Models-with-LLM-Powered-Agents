# Emergent-Recommendation-Systems-Combining-Agent-Based-Models-with-LLM-Powered-Agents

This project is a market simulation based on agent-based modeling (ABM), where investors with different profiles (conservative, moderate, aggressive) make investment decisions based on recommendations from two analysts, each powered by a local language model (LLM).

## Features
- Market data generation using **Geometric Brownian Motion (GBM)** to simulate realistic price histories.
- Investment decisions based on **LLaMA** and **Mistral** (local LLMs).
- Word overlap calculation to measure agreement between analysts.
- Logging of decisions and metrics in a CSV file (`market_results.csv`).
