import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py



train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_dataset = train_datagen.flow_from_directory(
    r"D:\Eye_data_set\train",
    target_size=(224, 224),
    batch_size=50,
    class_mode='binary'
)

test_dataset = validation_datagen.flow_from_directory(
    r"D:\Training2\dataset\test",
    target_size=(224, 224),
    batch_size=10,
    class_mode='binary'
)



print(train_dataset.class_indices)
print(test_dataset.class_indices)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


##history =
history=model.fit(train_dataset, epochs=10,validation_data=test_dataset)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()
##model_file = 'Model.h5'
model.save("Drowsy3.keras")

# Save the model to a file using pickle
##with open(model_file, 'wb') as f:
##    pickle.dump(model, f)

print("Model saved successfully!")



    
    
    

