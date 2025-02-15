# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install all required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the FastAPI server when the container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
