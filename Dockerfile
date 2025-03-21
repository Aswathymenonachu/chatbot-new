# Use a lightweight Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Run the chatbot
CMD ["streamlit", "run", "chatbot.py", "--server.port=8501", "--server.enableCORS=false"]
