# 1. Use an official, lightweight Python base image
FROM python:3.9-slim

# 2. Set a working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. Expose the port the app will run on
EXPOSE 8080

# 7. Define the command to start your Uvicorn server
CMD ["uvicorn", "scan:app", "--host", "0.0.0.0", "--port", "8080"]