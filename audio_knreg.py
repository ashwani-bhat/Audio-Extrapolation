# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 23:30:07 2018

@author: Ashwani
"""

import numpy as np
import pandas as pd
import winsound
import scipy.io.wavfile as wav

def PlaySound(name_of_file,sample_rate,data):
    wav.write(name_of_file + ".wav",sample_rate,data)
    winsound.PlaySound(name_of_file + ".wav",winsound.SND_FILENAME)

Provided_Portion = int(input("Enter The Percentage Input\t"))/100

audio_file=[]

# Loop through the dataset and load up all 50 of the 0_jackson*.wav
import os
for file in os.listdir('free-spoken-digit-dataset-master/recordings'):
    if file.startswith('0_jackson'):
        a=os.path.join('free-spoken-digit-dataset-master/recordings',file)
        sample_rate,audio_data = wav.read(a)
        audio_file.append(audio_data)
print(len(audio_file))

audio_file=pd.DataFrame(data=audio_file,dtype=np.int16)
audio_file.dropna(axis = 1,inplace=True)
audio_file=audio_file.values
print(type(audio_file))

n_audio_samples = audio_file.shape[1]
print(n_audio_samples)

#KNeighborsRegression
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)

from sklearn.utils.validation import check_random_state
rng   = check_random_state(7)  # Leave this alone until you've submitted your lab
random_idx = rng.randint(audio_file.shape[0])
test  = audio_file[random_idx]
train = np.delete(audio_file, [random_idx], axis=0)

portion = int(Provided_Portion * n_audio_samples)
X_train = train[:,:portion]
y_train = train[:,portion:]

model.fit(X_train,y_train)

# TESTING PHASE STARTS

sr, test = wav.read("ahsan_sir.wav")  # read the file
test = test[-train.shape[1]:]
test = test[:,0]
wav.write('Original Test Clip.wav', sr, test) # Write original test file without clipping

PlaySound("Original Test Clip",sr, test) # Play the original test file
portion = int(Provided_Portion * n_audio_samples)
X_test = test[:portion] 

PlaySound("x_test",sr,X_test)
y_test = test[portion:]

X_test = X_test.reshape(1, -1)
y_test = y_test.reshape(1, -1)

y_test_prediction = model.predict(X_test)
y_test_prediction = y_test_prediction.astype(dtype=np.int16)
#score = model.score(X_test,y_test_prediction)
#print("Extrapolation R^2 Score: ", score)
from sklearn import metrics
ae = metrics.mean_absolute_error(y_test,y_test_prediction)
print(ae)

completed_clip = np.hstack((X_test, y_test_prediction))
completed_clip = completed_clip.reshape(-1)
wav.write('Extrapolated Clip.wav', sr, completed_clip)

winsound.PlaySound('Extrapolated Clip.wav', winsound.SND_FILENAME)
winsound.PlaySound('Original Test Clip.wav', winsound.SND_FILENAME)

#Visualization of data
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(y_test.reshape(-1,1),color='red',label='Original')
plt.plot(y_test_prediction.reshape(-1,1),color='blue',label='Extrapolated',alpha=0.4)
plt.legend()
plt.show()

plt.scatter(y_test,y_test_prediction);

sns.violinplot(x=y_test_prediction,color="red",label="Prediction",alpha=0.4)
sns.violinplot(x=y_test,color="blue",label="Original")
plt.legend(["Prediction","Original"],cmap=["red","blue"])
plt.show()

sns.boxplot(x=y_test,hue=y_test_prediction)