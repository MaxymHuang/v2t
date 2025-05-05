# Use an official Python image as the base
FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy the requirements directly
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code into the container
COPY server.py .

# Expose the port Flask will use
EXPOSE 5000

# Run the application
CMD ["python", "server.py"]

