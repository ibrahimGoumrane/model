FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install system dependencies for face_recognition
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    dlib \
    face_recognition \
    scipy \
    pathlib \
    pickle-mixin

# Set the default command to run the application
CMD ["python", "detector.py"]