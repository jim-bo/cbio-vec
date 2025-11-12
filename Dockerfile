# Use an official PyTorch runtime as a parent image for better optimization
FROM pytorch/manylinux-cpu:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make port 8123 available to the world outside this container
EXPOSE 8123

# Define environment variable
ENV INDEX_DIR=./index_dir

# Run mcp_server.py when the container launches
CMD ["python", "mcp_server.py", "--host", "0.0.0.0", "--port", "8123"]
