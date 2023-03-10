import os

# Path to the waves folder
folder_path = 'datasets/not_waves/'

# Get all file names in the folder
file_names = os.listdir(folder_path)

# Rename all files with numbers
for i, file_name in enumerate(file_names):
    file_path = os.path.join(folder_path, file_name)
    new_file_name = str(i) + os.path.splitext(file_name)[1]
    new_file_path = os.path.join(folder_path, new_file_name)
    os.rename(file_path, new_file_path)
    
    

# # Set path to folder containing waves images
# folder_path = 'waves/'

# # Get list of all file names in folder
# file_names = os.listdir(folder_path)

# # Loop through all files in folder
# for i, file_name in enumerate(file_names):
#     # Get file extension
#     ext = os.path.splitext(file_name)[1]

#     # Check if file extension is not PNG
#     if ext != '.png':
#         # Set new file name as number with leading zeros and PNG extension
#         new_file_name = '{:03d}.png'.format(i+1)

#         # Rename file
#         os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))

#         # Convert image to PNG format
#         img = Image.open(os.path.join(folder_path, new_file_name))
#         img.save(os.path.join(folder_path, new_file_name))