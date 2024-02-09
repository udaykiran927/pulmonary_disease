import streamlit as st
import shutil
import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.preprocessing import LabelEncoder



def Filter_Denoised(raw_audio, sample_rate, filter_order, filter_lowcut, filter_highcut, btype="bandpass"):
    b, a = 0,0
    if btype == "bandpass":
        b, a = signal.butter(filter_order, [filter_lowcut/(sample_rate/2), filter_highcut/(sample_rate/2)], btype=btype)

    if btype == "highpass":
        b, a = signal.butter(filter_order, filter_lowcut, btype=btype, fs=sample_rate)

    audio = signal.lfilter(b, a, raw_audio)

    return audio



def build_spectogram(file_path):
    plt.interactive(False)
    file_audio_series,sr = librosa.load(file_path,sr=None)    
    spec_image = plt.figure(figsize=[1,1])
    ax = spec_image.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectogram = librosa.feature.melspectrogram(y=file_audio_series, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectogram, ref=np.max))
    
    image_name  = 'logmel/spectfile.png'
    plt.savefig(image_name, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    spec_image.clf()
    plt.close(spec_image)
    plt.close('all')







def main():
    st.title("Audio Upload and Display App")

    # Upload audio file
    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"])

    if audio_file is not None:
        # Display uploaded audio file
        st.audio(audio_file, format="audio/*")
        # Display message
        sample_rate = 4000
        filter_lowcut = 50
        filter_highcut = 1800
        filter_order = 5
        filter_btype = "bandpass"
        raw_audio, sample_rate = librosa.load(audio_file, sr=sample_rate)
        # Noise reduction method, filter
        audio_data = Filter_Denoised(raw_audio, sample_rate, filter_order,filter_lowcut,filter_highcut, btype=filter_btype)
        save_path='denoise/soundfile.wav'
        write(save_path, sample_rate, audio_data)
        build_spectogram(save_path)
        # Load the trained LSTM model
        model = load_model('lstm_all.h5')
        labels = np.load('labels.npy')
        # Load the VGG16 model without the top classification layer
        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Load and preprocess the test image
        test_image_path = 'logmel/spectfile.png'
        img = load_img(test_image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features from the test image using VGG16
        features = vgg16.predict(img_array)


        # Make prediction using the trained LSTM model
        prediction = model.predict(features.reshape((features.shape[0], -1, features.shape[-1])))

        # Convert prediction probabilities to class labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

        st.write(predicted_class)
        





        #st.success("Send chesthey hamaya ipoyindhi emo inka manaki em work undadu le ani happy ga feel avaku, Anni teams baga perform chesaru anta review-1 lo mana paristhithi alochinchandi.Entha sepu evaro okaru chestharu le ani relax avadam kadu, team leader chepey antha varaku em work undadu ani relax avadam kadu . me intrest tho next em cheyali ani start cheyadam nerchukondi..😡😡😡")

if __name__ == "__main__":
    main()
