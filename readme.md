A STEM course planning agent developed for students at BataGo, a renowned STEM education institute in Shanghai, leverages real-world data to enhance learning. The agent analyzes teaching records stored in Excel, aligns with individual career goals, and generates personalized 30-hour Project-Based Learning (PBL) plans.

This system utilizes Ollama and LangChain, with the qwen3-vl:235b-cloud model serving as the underlying large language model (LLM).

# Install

pip install -r requirements.txt

# run

python batago.py

# pre-request

ollama pull glm-4.6:cloud
ollama pull qwen3-vl:235b-cloud
