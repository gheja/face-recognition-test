#!/bin/env python3

import os
import numpy as np
import face_recognition

def load_known_faces():
	print("Loading known faces...")
	
	os.chdir(os.path.join(base_dir, "known_faces"))
	
	for root, dirs, files in os.walk('.'):
		for f in files:
			filename = os.path.join(root, f)
			
			print("  {} > {}".format(root, f))
			
			image = face_recognition.load_image_file(filename)
			
			# Search for all the faces
			face_locations = face_recognition.face_locations(image)
			
			# Pick the first one and get its encoding
			face_encodings = face_recognition.face_encodings(image, face_locations[0])
			
			# Store the processed face
			known_face_encodings.append(face_encodings[0])
			known_face_names.append(os.path.basename(root))

def recognize_faces():
	print("Detecting and recognizing faces...")
	
	os.chdir(os.path.join(base_dir, "test_images"))
	
	for root, dirs, files in os.walk('.'):
		for f in files:
			filename = os.path.join(root, f)
			
			image = face_recognition.load_image_file(filename)
			
			print("  {}: processing...".format(filename))
			
			# Search for all the faces
			face_locations = face_recognition.face_locations(image)
			
			# Then calculate their encodings
			face_encodings = face_recognition.face_encodings(image, face_locations)
			
			# Loop through each face found in the unknown image
			for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
				name = "Unrecognized"
				confidence = 0
				
				# Find the distance of the face to every known faces. Distance is 0.0 at an exact match
				# and gets higher with the distance, can be more than 1.0
				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				
				# Use the known face with the smallest distance to the new face
				best_match_index = np.argmin(face_distances)
				
				# If there is a possible match
				if face_distances[best_match_index] < 0.6:
					name = known_face_names[best_match_index]
					
					# Confidence is the opposite of distance
					confidence = 1 - face_distances[best_match_index]
				
				print("    found {} with confidence {:.2f}".format(name, confidence))

base_dir = os.getcwd()

known_face_encodings = []
known_face_names = []

load_known_faces()
recognize_faces()
