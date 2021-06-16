
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import Callback
from keras.models import load_model


#---------------------------------------------------------------------

# DATA PREPROCESSING

dataset_dir = r"C:/Users/dipesh/Desktop/Urban8K_Dataset"
df = pd.read_csv(dataset_dir + r"/UrbanSound8K.csv")

# I'll train on 1000 samples only to avoid running out of memory.
df = df.sample(n=1000)

num_samples = 1000

#---------------------------------------------------------------------

# Takes audio file name/path & returns its Mel Spectrogram.
def convert_audio2MelSpec(audio_file):
    samples, sample_rate = librosa.load(audio_file, sr=None)
    spectrogram = librosa.stft(samples)
    sgram_mag, _ = librosa.magphase(spectrogram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
    mel_spectrogram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    return mel_spectrogram


# Takes 2D array & desired_shape then pads it with 0 to reshape it to desired shape.
def apply_padding(an_array, desired_shape):
    shape = np.shape(an_array)
    # we'll reshape all mel_spec to largest shape present in our dataset-(128, 1501)
    padded_array = np.zeros(desired_shape)
    padded_array[:shape[0], :shape[1]] = an_array
    return padded_array

#---------------------------------------------------------------------

# Finding the largest shape of mel_spectrogram in our samples.
# Since we are converting audio to mel_spec,the no. of rows is common in all i.e 128
# so we just need to find the largest no. of columns.

largest_shape = [128, 0]
i = 0
for audio_filename, fold, label in df[["slice_file_name", "fold", "class"]].values:
    file_path = dataset_dir + "/" + "fold" + str(fold) + "/" + audio_filename
    mel_spec = convert_audio2MelSpec(file_path)
    mel_shape = list(mel_spec.shape)
    if mel_shape[1] > largest_shape[1]:
        largest_shape[1] = mel_shape[1]
    i += 1

    # prints percentage of task completed.
    if i % 200 == 0:
        percent = (i / num_samples) * 100
        print("{} % Task Completed...".format(round(percent, 2)))

largest_shape = tuple(largest_shape)
print("Largest Shape : ", largest_shape)

#---------------------------------------------------------------------

x = []
y = []
i = 0
for audio_filename, fold, label in df[["slice_file_name", "fold", "class"]].values:

    file_path = dataset_dir + "/" + "fold" + str(fold) + "/" + audio_filename
    mel_spec = convert_audio2MelSpec(file_path)
    # padding to largest shape in dataset
    mel_spec = apply_padding(mel_spec, largest_shape)
    # converting 2D numpy array to 2D list
    mel_spec = mel_spec.tolist()
    x.append(mel_spec)
    y.append(label)
    i += 1
    if i % 300 == 0:
        print("{} samples processed...".format(i))

x = np.array(x)
y = np.array(y)

print(x[:3])
print(y[:3])

#---------------------------------------------------------------------

y_df = pd.DataFrame(data=y)

# contains all string class labels
labels = (list(pd.get_dummies(y_df)))

# one-hot-encoding labels
y = pd.get_dummies(y).values

print(labels)
print(y[:3])

np.save("X_Urban8K", x)
np.save("Y_Urban8K", y)

#---------------------------------------------------------------------

"""
Conv2D however expects 4 dimensions,because it also expects the channels dimension of image,
which in MNIST is nonexistent because it’s grayscale data and hence is 1.
Reshaping the data, while explicitly adding the channels dimension, resolves the issue.
The input shape a CNN accepts should be in a specific format.
In Tensorflow,the format is (num_samples, height, width, channels)
"""
print("Before Reshaping : ", x.shape)
largest_shape = list(largest_shape)
x = x.reshape(x.shape[0], largest_shape[0], largest_shape[1], 1)
print("After Reshaping", x.shape)

# Splitting data into train & test
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=0)


#---------------------------------------------------------------------

# MODEL TRAINING

input_shape = (x.shape[1], x.shape[2], x.shape[3])

# Defining the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), padding="same", input_shape=input_shape))
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=(2, 2), padding="same"))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=(2, 2), padding="same"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(150, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(len(y_train[0]), activation="softmax"))

print("Y_train length :", len(y_train[0]))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

#---------------------------------------------------------------------

# Custom Keras callback to stop training when certain accuracy is achieved.
class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


model.fit(x_train, y_train, epochs=50,
          callbacks=[MyThresholdCallback(0.9)], validation_data=(x_test, y_test))

#---------------------------------------------------------------------

# Saving the model
model.save("Audio_Classification_CNN")


