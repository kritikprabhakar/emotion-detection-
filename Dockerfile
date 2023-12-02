# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and model files into the container

COPY . .

# Install necessary dependencies
RUN pip install -r requirements.txt

# Run the Python script when the container launches
CMD ["python", "EmotionDetector.py"]

