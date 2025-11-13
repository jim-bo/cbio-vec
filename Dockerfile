# Use a slim base image
FROM python:3.12-slim

# Install PyTorch CPU only
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY index_dir /app/index_dir
COPY mcp_server.py /app/mcp_server.py

# Make port 8123 available to the world outside this container
EXPOSE 8123

# Define environment variable
ENV INDEX_DIR=./index_dir

# Run mcp_server.py when the container launches
CMD ["python", "mcp_server.py", "--host", "0.0.0.0", "--port", "8123"]
