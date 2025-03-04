# Face Recognition Model

## Overview
This project is a face recognition application that uses the `face_recognition` library to encode and recognize faces from images.

## Project Structure


## Prerequisites
- Docker
- Docker Compose

## Steps to Run the Project on Docker

1. **Clone the Repository**
   ```sh
   git clone <repository-url> 
   cd <repository-directory>

2. **Run and build the docker Container**   
    ```docker-compose up --build```

This command will build the Docker image and start the container. The application will run the detector.py script inside the container.

## App Explication 

the main file used here is **detector.py** : 

### `encode_known_faces`
This function is responsible for encoding known faces from the training dataset.

### `recognize_faces`
This function is used to recognize faces in a given image


## Example Usage 
```sh 
face_handler = FaceRecognitionHandler()
# face_handler.encode_known_faces()

training_path = Path("validation/mandy.jpg")
print(face_handler.recognize_faces(training_path, 'cp', '1', 'A'))
