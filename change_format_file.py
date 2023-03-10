from PIL import Image
import os

input_folder = 'datasets/not_waves'
output_folder = 'datasets/not_waves'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # Open the image file
        img = Image.open(os.path.join(input_folder, filename))

        # Convert the image to RGB format if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save the image as a jpg file
        jpg_filename = os.path.splitext(filename)[0] + '.jpg'
        img.save(os.path.join(output_folder, jpg_filename), 'JPEG')
