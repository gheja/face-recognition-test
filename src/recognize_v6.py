#!/bin/env python3

import os
import numpy as np
import face_recognition
import PIL
import PIL.ImageDraw
from sklearn.cluster import DBSCAN
import pickle

# Load an image, resize it to 1200x1200 if larger, convert it to RGB, return it as a numpy array
def load_image(file):
	image = PIL.Image.open(file)
	
	image.thumbnail((1200, 1200))
	
	image = image.convert('RGB')
	
	return np.array(image)

def load_known_faces():
	print("Loading known faces...")
	
	os.chdir(os.path.join(base_dir, "known_faces"))
	
	for root, dirs, files in os.walk('.'):
		for f in files:
			if f == ".placeholder":
				continue
			
			filename = os.path.join(root, f)
			
			print("  {} > {}".format(root, f))
			
			# Get a filename for the cached data
			cache_filename = os.path.join(base_dir, "known_faces_cache", root, f + ".pickle")
			
			# Read cached data if exists
			if os.path.exists(cache_filename):
				face_encodings = pickle.load(open(cache_filename, 'rb'))
			else:
				# Create the cache directory if does not exist
				if not os.path.exists(os.path.dirname(cache_filename)):
					os.makedirs(os.path.dirname(cache_filename))
				
				try:
					image = load_image(filename)
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
				
				# Save data to the cache
				pickle.dump(face_encodings, open(cache_filename, 'wb'))
			
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
				image = load_image(filename)
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
				
				if name == "Unrecognized":
					unrecognized_faces.append({
						"face_location": (top, right, bottom, left),
						"face_encoding": face_encoding,
						"filename": filename
					})
			
			# Remove the drawing library from memory as per the Pillow docs
			del draw
			
			# Save the annotated image
			outputName = os.path.join(base_dir, "annotated_images", f)
			image2.save(outputName)

def cluster_unrecognized_faces():
	print("Clustering unrecognized faces...")
	
	if len(unrecognized_faces) == 0:
		return
	
	encodings = []
	
	# Create an array suitable for DBSCAN conaining only the encodings
	for face in unrecognized_faces:
		encodings.append(face["face_encoding"])
	
	# Group at least 3 faces ("min_samples") with a maximal distance
	# of 0.5 ("eps"). From a few runs these numbers seems to be working
	# well for me. Your case may vary so try to tune these if you don't
	# like your results.
	#
	# For more info see:
	#   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
	clt = DBSCAN(metric="euclidean", n_jobs=-1, min_samples=3, eps=0.5)
	clt.fit(encodings)
	
	print("Saving clustered faces...")
	
	face_index = 0
	for face in unrecognized_faces:
		person_index = clt.labels_[face_index]
		
		if person_index == -1:
			name = "unclustered"
		else:
			name = "person_%d" % (person_index)
		
		filename = os.path.join(base_dir, "test_images", face["filename"])
		clustered_filename = os.path.join(base_dir, "clustered_faces", name, "face_%d.png" % (face_index))
		
		# Create the directory for this person if needed
		if not os.path.exists(os.path.dirname(clustered_filename)):
			os.makedirs(os.path.dirname(clustered_filename))
		
		try:
			image = load_image(filename)
		except:
			print("{}: failed to load, skipping.".format(filename))
			continue
		
		(top, right, bottom, left) = face["face_location"]
		
		# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
		# See http://pillow.readthedocs.io/ for more about PIL/Pillow
		image2 = PIL.Image.fromarray(image)
		
		# Crop the image to the face
		image2 = image2.crop((left, top, right, bottom))
		
		image2.save(clustered_filename)
		
		face_index += 1

base_dir = os.getcwd()

known_face_encodings = []
known_face_names = []
unrecognized_faces = []

load_known_faces()
recognize_faces()
cluster_unrecognized_faces()
