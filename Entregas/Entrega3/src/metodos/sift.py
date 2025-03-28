import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


def precompute(image):
    """
    Precomputes the SIFT keypoints and descriptors for the given image.
    
    Args:
        image (numpy.ndarray): The input image for which to compute features.
    
    Returns:
        tuple: A tuple containing keypoints and descriptors.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def compare(image, model_features):
    """
    Compares the SIFT features of the input image with the precomputed model features.
    
    Args:
        image (numpy.ndarray): The input image to compare.
        model_features (tuple): A tuple containing keypoints and descriptors of the model image.
    
    Returns:
        float: The similarity score based on the number of good matches.
    """
    model_keypoints, model_descriptors = model_features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if descriptors is None or model_descriptors is None:
        return 0  # No descriptors to compare

    # Use BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors, model_descriptors)

    # Sort matches by distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Define a threshold for "good" matches
    good_matches = [m for m in matches if m.distance < 0.75 * matches[-1].distance]

    # Return the number of good matches as the similarity score
    return len(good_matches)
