#!/bin/env python3

import os
import numpy as np
import face_recognition
import PIL
import PIL.ImageDraw

def load_known_faces():
	print("Loading known faces...")
	
	os.chdir(os.path.join(base_dir, "known_faces"))
	
	for root, dirs, files in os.walk('.'):
		for f in files:
			if f == ".placeholder":
				continue
			
			filename = os.path.join(root, f)
			
			print("  {} > {}".format(root, f))
			
			try:
				image = face_recognition.load_image_file(filename)
			except:
				print("  {}: failed to load, skipping.".format(filename))
				continue
			
			if image is None:
				print("  {}: failed to load, skipping.".format(filename))
				continue
			
			# Search for all the faces
			face_locations = face_recognition.face_locations(image)
			
			if len(face_locations) == 0:
				print("  {}: no faces found, skipping.".format(filename))
				continue
			
			# Pick the first one and get its encoding
			face_encodings = face_recognition.face_encodings(image, face_locations[0])
			
			if face_encodings is None:
				print("  {}: failed to process, skipping.".format(filename))
				continue
			
			if len(face_encodings) == 0:
				print("  {}: processed but empty, skipping.".format(filename))
				continue
			
			# Store the processed face
			known_face_encodings.append(face_encodings[0])
			known_face_names.append(os.path.basename(root))

def recognize_faces():
	print("Detecting and recognizing faces...")
	
	os.chdir(os.path.join(base_dir, "test_images"))
	
	for root, dirs, files in os.walk('.'):
		for f in files:
			if f == ".placeholder":
				continue
			
			filename = os.path.join(root, f)
			
			try:
				image = face_recognition.load_image_file(filename)
			except:
				print("  {}: failed to load, skipping.".format(filename))
				continue
			
			if image is None:
				print("  {}: failed to load, skipping.".format(filename))
				continue
			
			print("  {}: processing...".format(filename))
			
			# Search for all the faces
			face_locations = face_recognition.face_locations(image)
			
			if len(face_locations) == 0:
				print("  {}: no faces found, skipping.".format(filename))
				continue
			
			# Then calculate their encodings
			face_encodings = face_recognition.face_encodings(image, face_locations)
			
			# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
			# See http://pillow.readthedocs.io/ for more about PIL/Pillow
			image2 = PIL.Image.fromarray(image)
			
			# Create a Draw instance to draw to
			draw = PIL.ImageDraw.Draw(image2)
			
			# Loop through each face found in the unknown image
			for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
				name = "Unrecognized"
				confidence = 0
				
				if len(known_face_encodings) > 0:
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
				
				if confidence >= 0.5:
					color = (0, 190, 0)
					color2 = (255, 255, 255)
				elif confidence >= 0.35:
					color = (0, 120, 190)
					color2 = (255, 255, 255)
				elif confidence >= 0.25:
					color = (0, 0, 120)
					color2 = (255, 255, 255)
				else:
					color = (30, 30, 30)
					color2 = (90, 90, 90)
				
				# Draw a box around the face
				draw.rectangle(((left, top), (right, bottom)), outline = color)
				
				# Draw a label with the name below the face
				text_width, text_height = draw.textsize(name)
				draw.rectangle(((left, bottom), (right, bottom + text_height)), fill = color, outline = color)
				draw.text((left + 6, bottom), "{:.2f} {}".format(confidence, name), fill = color2)
			
			
			# Remove the drawing library from memory as per the Pillow docs
			del draw
			
			# Save the annotated image
			outputName = os.path.join(base_dir, "annotated_images", f)
			image2.save(outputName)

base_dir = os.getcwd()

known_face_encodings = []
known_face_names = []

load_known_faces()
recognize_faces()
