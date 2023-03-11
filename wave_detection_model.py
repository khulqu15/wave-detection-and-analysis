import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Define callback to plot transfer function and save image after each epoch
class PlotCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['val_loss'])
        self.accs.append(logs['val_accuracy'])

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].plot(self.losses)
        ax[0].set_title('Validation Loss')

        ax[1].plot(self.accs)
        ax[1].set_title('Validation Accuracy')

        if self.validation_data is not None:
            ax[0].figure.savefig('val_loss_{:03d}.png'.format(epoch))
            ax[0].clear()
            ax[1].figure.savefig('val_acc_{:03d}.png'.format(epoch))
            ax[1].clear()

            plt.imshow(self.validation_data[0][0], cmap='gray')
            plt.savefig('validation_data_{:03d}.png'.format(epoch))
            plt.clf()

# Load dataset
data_path = 'datasets/'
categories = ['waves', 'not_waves']
data = []
label = []

for category in categories:
    path = os.path.join(data_path, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (150, 150))
        data.append(img_arr)
        label.append(class_num)

# Reshape data
data = np.array(data) / 255.0
data = np.reshape(data, (-1, 150, 150, 1))

# Split dataset into training and validation sets
label = tf.keras.utils.to_categorical(label)
train_data = data[:int(len(data)*0.8)]
train_label = label[:int(len(label)*0.8)]
val_data = data[int(len(data)*0.8):]
val_label = label[int(len(label)*0.8):]

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((150, 150, 1), input_shape=(150, 150)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(categories), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
plot_callback = PlotCallback()
history = model.fit(train_data, train_label,
                    epochs=10,
                    validation_data=(val_data, val_label),
                    callbacks=[plot_callback])

# Evaluate model
test_data = []
test_label = []

for category in categories:
    path = os.path.join(data_path, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (150, 150))
        img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
        img_arr = np.array(img_arr) / 255.0
        test_data.append(img_arr)
        test_label.append(class_num)

test_data = np.array(test_data) / 255.0
test_label = tf.keras.utils.to_categorical(test_label)
test_loss, test_acc = model.evaluate(test_data, test_label)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Save model
model.save('wave_detection_model')

# Plot transfer function
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = 1/(1 + np.exp(-x))

plt.plot(x, y)
plt.title('Sigmoid Transfer Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()

# Save training images
for i in range(len(train_data)):
    img = train_data[i] * 255.0
    img = np.array(img, dtype=np.uint8)
    label = np.argmax(train_label[i])
    category = categories[label]
    filename = f'train_img_{i}_category_{category}.png'
    cv2.imwrite(filename, img)

