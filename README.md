# Deep Research AI System

A web-based research tool that crawls websites using Tavily, processes data with a dual-agent system (Research and Drafting agents), and generates structured answers via the Hugging Face Inference API. Built with LangGraph and LangChain, deployed on Streamlit Community Cloud.

## Overview

This project fulfills the assignment:  
*"Design a Deep Research AI Agentic System that crawls websites using Tavily for online information gathering. Implement a dual-agent (or more agents) system with one agent focused on research and data collection, while the second agent functions as an answer drafter. The system should utilize the LangGraph & LangChain frameworks to effectively organize the gathered information."*

### Features
- **Web Crawling**: Uses Tavily API to gather web data.
- **Dual-Agent System**: Research Agent collects data; Drafting Agent generates answers.
- **Frameworks**: LangGraph for workflow, LangChain for tools.
- **Custom UI**: Streamlit interface with expandable research data (preview + full text).
- **Deployment**: Hosted on Streamlit Community Cloud via GitHub.

## Prerequisites
- **Python**: 3.9 or higher
- **API Keys**:
  - Hugging Face: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free tier: 1,000 requests/day)
  - Tavily: [tavily.com](https://tavily.com/)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/T095T/DeepCrawl.git
2. **Install Dependencies**:
   pip install -r requirements.txt

## Running the system
streamlit run app.py
