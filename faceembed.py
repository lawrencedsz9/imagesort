import cv2
import numpy as np
import os
import shutil
import pickle
import dlib
from sklearn.metrics.pairwise import cosine_similarity

# Paths configuration - adjust these to your Windows paths
input_image_path = 'selfie.jpg'  # Your selfie
dataset_folder = r'C:\Users\lawre\OneDrive\Documents\inter\dataset'  # Folder with all photos to search through
output_folder = 'matched_images'  # Where to save matches
index_file = 'faces_index.pkl'  # Cache of face embeddings

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load face detection and recognition models
print("Loading face models...")
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def detect_faces(image):
    """Detect all faces in an image and return their bounding boxes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_detector(gray)

def get_face_embedding(image, face):
    """Create embedding for a single face using Dlib's face recognition model."""
    shape = shape_predictor(image, face)
    embedding = face_recognition_model.compute_face_descriptor(image, shape)
    return np.array(embedding)

def index_faces(dataset_path):
    """Index all faces from the dataset and store embeddings with image paths."""
    print(f"Indexing faces from {dataset_path}...")
    index = []
    total_files = len(os.listdir(dataset_path))
    processed = 0
    
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            # Detect faces
            faces = detect_faces(img)
            
            # Process each face
            for face in faces:
                embedding = get_face_embedding(img, face)
                index.append((embedding, img_path))
                
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{total_files} images")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Save index
    with open(index_file, 'wb') as f:
        pickle.dump(index, f)
    print(f"Indexed {len(index)} faces from {processed} images.")
    return index

def find_matches(selfie_path, similarity_threshold=0.6):
    """Find matching images based on similarity to faces in the selfie."""
    # Check if index exists, if not create it
    if not os.path.exists(index_file):
        print("Face index not found. Creating index...")
        index = index_faces(dataset_folder)
    else:
        # Load existing index
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        print(f"Loaded index with {len(index)} faces.")
    
    # Process the selfie
    print(f"Processing selfie: {selfie_path}")
    selfie_img = cv2.imread(selfie_path)
    if selfie_img is None:
        print(f"Error: Could not read selfie at {selfie_path}")
        return []
        
    selfie_faces = detect_faces(selfie_img)
    
    if len(selfie_faces) == 0:
        print("No faces detected in the selfie.")
        return []
    
    print(f"Detected {len(selfie_faces)} faces in the selfie.")
    
    # Get embeddings for each face in the selfie
    selfie_embeddings = []
    for face in selfie_faces:
        embedding = get_face_embedding(selfie_img, face)
        selfie_embeddings.append(embedding)
    
    # Find matches for each face in the selfie
    matched_paths = set()
    
    for selfie_embedding in selfie_embeddings:
        for db_embedding, img_path in index:
            # Calculate cosine similarity
            similarity = cosine_similarity(
                selfie_embedding.reshape(1, -1), 
                db_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > similarity_threshold:
                matched_paths.add(img_path)
                
    print(f"Found {len(matched_paths)} matching images.")
    return matched_paths

def copy_matched_images(matched_paths, destination_folder):
    """Copy matched images to the output folder."""
    count = 0
    for img_path in matched_paths:
        try:
            img_name = os.path.basename(img_path)
            output_path = os.path.join(destination_folder, img_name)
            shutil.copy(img_path, output_path)
            count += 1
        except Exception as e:
            print(f"Error copying {img_path}: {str(e)}")
    
    print(f"Copied {count} matched images to {destination_folder}")

def main():
    """Main function to execute the face matching workflow."""
    print("==== Face Matching System ====")
    
    # Step 1: Find matches
    matched_paths = find_matches(input_image_path)
    
    # Step 2: Copy matched images to output folder
    if matched_paths:
        copy_matched_images(matched_paths, output_folder)
        print(f"All done! Check {output_folder} for your matches.")
    else:
        print("No matches found.")

if __name__ == "__main__":
    main()