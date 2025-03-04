from pathlib import Path
import face_recognition
import pickle
from collections import Counter
import numpy as np
import os
from scipy.spatial.distance import euclidean


DEFAULT_ENCODINGS_PATH = Path("encoding")
DEFAULT_TRAINING_PATH=Path('training')
DEFAULT_VALIDATION_PATH=Path('validation')



DEFAULT_TRAINING_PATH.mkdir(exist_ok=True)
DEFAULT_ENCODINGS_PATH.mkdir(exist_ok=True)
DEFAULT_VALIDATION_PATH.mkdir(exist_ok=True)



class imageException(Exception):
    def __init__(self, errorImage) -> None:
        super(imageException,self).__init__(errorImage)


class FaceRecognitionHandler:
    def __init__(self, encodings_location=DEFAULT_ENCODINGS_PATH):
        self.encodings_location = encodings_location
        self.model = "CNN"

    def __recognize_face(self,unknown_encoding, reference_encoding):
        """"
        This function is used to recognize the face of the person in the image by comparing the unknown encoding with the reference encoding
        Args:
            unknown_encoding: The encoding of the person in the image
            reference_encoding: The encoding of the person in the reference image
        Returns the name of the person if the face is recognized
        """

        boolean_matches = face_recognition.compare_faces(reference_encoding['encodings'], unknown_encoding)
        votes = Counter(
            name
            for match, name in zip(boolean_matches, reference_encoding["names"])
            if match
        )
        if votes:
            name = votes.most_common(1)[0][0]
            ele_len = name.rfind('_encodings')
            return name[:ele_len]


    def __save_encodings(self,relative_path , filename ,name_encodings):
        """
        This function is used to save the encodings of the person in the image
        Args:
            relative_path: The relative path of the file
            filename: The name of the file
            name_encodings: The encodings of the person in the image
        """


        if not filename:
            return
        
        # Saving the encodings to a file with the name of the person
        with Path(relative_path).joinpath(f"{filename}.pkl").open(mode="wb") as f:
            pickle.dump(name_encodings, f)

    def __handle_encodings(self, relative_path , filename, show_file_error=True):
        """
        This function is used to load the encodings of the person in the image
        Args:
            relative_path: The relative path of the file
            filename: The name of the file
            show_file_error: If the file error is to be shown
        Returns the encodings of the person in the image
        """


        old_encodings = self.__load_encoded_faces(relative_path , filename, show_file_error == True)
        if old_encodings:
            return old_encodings
        return {'names': [], 'encodings': []}

    def __load_encoded_faces(self, relative_path , filename, show_file_error=True):
        try:
            with Path(relative_path).joinpath(f"{filename}.pkl").open(mode="rb") as f:
                loaded_encodings = pickle.load(f)
        except OSError as e:
            if show_file_error:
                print(f"An IOError occurred: {e}")
            return {'names': [], 'encodings': []}
        return loaded_encodings


    def __encoding_location(self,filepath:Path):
        """
        This function is used to get the relative path and the file name
        Args:
            filepath: The path of the file
            Returns the relative path and the file name
        
        """

        heirarcy_list=filepath.parent.name.split('_')  
        # File name in form of (Lastname,Firstname) at the end of the path
        filename='_'.join(heirarcy_list[-2:]) + '_encodings'
        # I want all the elements except the last two elements
        relative_path_list=[str(DEFAULT_ENCODINGS_PATH)] + heirarcy_list[:-2]
        # Joining the elements to get the relative path
        relative_path = '/'.join(relative_path_list)

        # make it lower case
        relative_path = relative_path.lower()

        # Returning the relative path and the file name
        return relative_path ,filename

    def __is_new_encoding(self, unique_encodings, new_encoding, threshold=0.5):
        """
        This function is used to check if the encoding is new
        Args:
            unique_encodings: The unique encodings
            new_encoding: The new encoding
            threshold: The threshold value
        Returns True if the encoding is new
        """
        for old_encoding in unique_encodings:
            if euclidean(old_encoding, new_encoding) < threshold:  # If too similar
                return False
        return True
    
    def encode_known_faces(self):
        """
        This function is used to encode the known faces
        """

        # Iterating through the training path
        for filepath in DEFAULT_TRAINING_PATH.glob("*/*"):

            # creating the heirarchy
            relative_path , filename=self.__encoding_location(filepath)

            if not os.path.exists(relative_path): 
                # if the demo_folder directory is not present  
                # then create it. 
                os.makedirs(relative_path)    

            # Loading the image    
            image = face_recognition.load_image_file(filepath)

            # Getting the face encodings
            face_encodings_old = self.__handle_encodings(relative_path , filename, show_file_error=False)
            
            # Getting the face locations
            face_locations = face_recognition.face_locations(image, model=self.model)
            
            # Getting the face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if face_encodings_old['encodings']:
                
                for encoding in face_encodings:
                    if self.__is_new_encoding(face_encodings_old['encodings'], encoding):
                        face_encodings_old['encodings'].append(encoding)
                face_encodings=face_encodings_old['encodings']

            # Saving the encodings
            name_encodings = {"names": [filename], "encodings": face_encodings}
            self.__save_encodings(relative_path , filename ,name_encodings)


    def recognize_faces(self,image_location , *args ):
        """
        This function is used to recognize the faces in the image
        Args:
            image_location: The location of the image
            args: The arguments which are Where the student should be searched
        Returns the present people in the image
        """
        present_people=[]

        # Loading the image
        input_image = face_recognition.load_image_file(image_location)

        # Getting the face locations
        input_face_locations = face_recognition.face_locations(input_image, model=self.model)\
        
        # Getting the face encodings
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        if (not input_face_encodings) :
            print('the image entered is not clear, enter a clear image to recognize face ')
            raise imageException('Image not Clear')


        # Get the path based on the args
        relative_path = '/'.join([str(DEFAULT_ENCODINGS_PATH)] + list(args))

        # make it lower case
        relative_path = relative_path.lower()
        
        for files in Path(relative_path).glob('*_encodings.pkl'):
            # Get the name of the file
            file_name = files.stem

            # Load the encodings
            loaded_encodings = self.__load_encoded_faces(relative_path , file_name)
            if not loaded_encodings['encodings']:
                continue  # Skip if there are no encodings

            # for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
            for unknown_encoding in  input_face_encodings:
                searched_name = self.__recognize_face(unknown_encoding, loaded_encodings)
                if searched_name:
                    present_people.append(searched_name)
                    break
        return present_people




face_handler = FaceRecognitionHandler()
# face_handler.encode_known_faces()



training_path = Path("validation/mandy.jpg")
# training_path = Path("validation/ben_afflek_2.jpg")
# training_path = Path("./validation/elon.jpg")



print(face_handler.recognize_faces(training_path , 'cp', '1', 'A'))