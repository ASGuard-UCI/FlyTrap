## crop image


import numpy as np
import cv2
from PIL import Image


def crop_and_resize_image(image_path, output_path, size=300):
    # Load the image
    image = cv2.imread(image_path)
    
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Determine the shorter edge
    shorter_edge = min(height, width)
    
    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2
    
    # Calculate the coordinates for cropping
    crop_x1 = center_x - shorter_edge // 2
    crop_x2 = center_x + shorter_edge // 2
    crop_y1 = center_y - shorter_edge // 2
    crop_y2 = center_y + shorter_edge // 2
    
    # Crop the image to a square
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Resize the cropped image to the desired size
    resized_image = cv2.resize(cropped_image, (size, size))
    
    # Save the output image
    cv2.imwrite(output_path, resized_image)


def rotate_image(input_path, output_path, angle=135):
    # Open the image
    image = Image.open(input_path).convert("RGBA")
    
    # Get the size of the image
    width, height = image.size
    
    # Create a new blank image with the same size and a transparent background
    rotated_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Rotate the original image
    rotated = image.rotate(-angle, resample=Image.BICUBIC, expand=True)
    
    # Convert rotated image to "RGBA" to ensure it has an alpha channel
    rotated = rotated.convert("RGBA")
    
    # Calculate the position to paste the rotated image onto the blank image
    x = (width - rotated.width) // 2
    y = (height - rotated.height) // 2
    
    # Paste the rotated image onto the blank image
    rotated_image.paste(rotated, (x, y), rotated)
    
    # Save the rotated image
    rotated_image.save(output_path)
    print(f"Image saved to {output_path}")


def crop_to_circle(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Ensure the image is a square
    height, width = image.shape[:2]
    assert height == width, "Input image must be a square"
    
    # Create a mask with a white filled circle in the center
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = width // 2
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
    
    # Apply the mask to the image
    if image.shape[2] == 4:  # If the image has an alpha channel
        masked_image = cv2.bitwise_and(image, image, mask=mask)
    else:  # If the image doesn't have an alpha channel, add one
        b, g, r = cv2.split(image)
        masked_image = cv2.merge((b, g, r, mask))
    
    # Save the output image
    cv2.imwrite(output_path, masked_image)


def mixup_images(image_path1, image_path2, alpha):
    # Read the two images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    
    # Ensure the images have the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions to mixup.")
    
    # Blend the images using the given alpha
    blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    
    return blended_image


if __name__ == "__main__":
    # Example usage
    # crop_and_resize_image('umbrella/patch.png', 'umbrella/patch.png')
    # crop_to_circle('umbrella/patch.png', 'umbrella/patch.png')
    # rotate_image('umbrella/patch.png', 'umbrella/patch.png', angle=135)
    img = mixup_images('data/dataset_v4.0/train_template/person4_parkinglot2_instance1/0002.jpg', 
                       'data/dataset_v4.0/train_template/person2_grass1_instance1/0002.jpg', 
                       alpha=0.5)
    cv2.imwrite('debug.png', img)