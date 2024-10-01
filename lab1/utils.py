# utils.py
import pygame

def load_image(path, size=(20, 20)):
    try:
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, size)
        return image
    except pygame.error as e:
        print(f"Unable to load image at {path}: {e}")
        return None
