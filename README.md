# Face Recognition Component for Image Retrieval Application

This is the core machine learning logic component for a face recognition-based image retrieval system being developed for a photography website application.

**Note: This repository contains only the ML/computer vision logic. Integration with AWS S3 and the web application is being developed separately.**

## Overview

This component handles the fundamental face recognition tasks within the larger project:
1. Detect faces in uploaded images
2. Create facial embeddings (numerical representations of facial features)
3. Compare facial embeddings to find matching faces across the image database
4. Identify images containing the same individual

## Features

- Face detection using Dlib's frontal face detector
- Facial landmark detection with a 68-point predictor
- Face recognition using deep learning models
- Cosine similarity calculation for face matching
- Caching of face embeddings to improve retrieval performance

## Prerequisites

- Python 3.6+
- OpenCV
- NumPy
- Dlib
- scikit-learn
- Required model files:
  - `shape_predictor_68_face_landmarks.dat` (for facial landmark detection)
  - `dlib_face_recognition_resnet_model_v1.dat` (for face embedding extraction)

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install opencv-python numpy dlib scikit-learn
   ```
3. Download the required model files if not already present:
   - [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

## Current Development Usage

1. Place your selfie or reference image in the project root directory as `selfie.jpg`
2. Put sample images you want to search through in the `dataset` folder
3. Run the main script:
   ```
   python faceembed.py
   ```
4. Check the `matched_images` folder for all photos containing faces that match your selfie

## Configuration

You can modify the following parameters in `faceembed.py`:
- `input_image_path`: Path to your selfie or reference image
- `dataset_folder`: Directory containing sample photos to search through
- `output_folder`: Directory where matching images will be copied
- `similarity_threshold`: Adjust this value (0-1) to control matching sensitivity (default: 0.6)

## How It Works

1. **Indexing Phase**: The algorithm processes all images in the dataset and creates facial embeddings for each detected face.
   - These embeddings are stored in `faces_index.pkl` for faster subsequent retrievals
   - Each embedding is a 128-dimensional vector representing facial features

2. **Matching Phase**: When provided with a reference image, the system creates embeddings for all faces detected.
   - It then compares these embeddings with all embeddings in the index
   - If the cosine similarity exceeds the threshold, the image is considered a match

3. **Retrieval Phase**: All matched images are identified for further processing.

## Project Context

This ML component is part of a larger project for a photography website that will:
- Store and retrieve images from AWS S3
- Provide a web interface for users
- Allow personalized face recognition for registered users
- Optimize image retrieval through cloud infrastructure

## Future Integration Plans

In the complete application, this component will be:
- Connected to AWS S3 for image storage and retrieval
- Integrated with a web application backend
- Enhanced with additional features like LBPH (Local Binary Patterns Histograms)
- Optimized for production performance

## License

This project uses Dlib which is licensed under the Boost Software License 1.0. 