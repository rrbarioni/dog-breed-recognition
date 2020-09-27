# Set base image (host OS)
FROM python:3.7

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src and models directory to the working directory
COPY src/ ./src
COPY models/ ./models

# Command to run on container start
ENV batch_size=1
CMD [ "sh", "-c", "python ./src/app.py --batch_size $batch_size"]