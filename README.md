# Emotion recognize

This is my thesis project at University. Training a Deep Learning model Convolutional Neural Network (CNNs) to recognize emotion on the faces.
Best compatible with Python 2.7 on Ubuntu 16.04.

# Result
The CNN model can recognize 5 kinds of emotion: happy, sad, neutral, surprise, angry. 

![Emotion surprise](https://github.com/thanhtinhvan/Emotion-recognize/blob/master/Screenshoots/supprise.png)

![Emotion angry](https://github.com/thanhtinhvan/Emotion-recognize/blob/master/Screenshoots/angry.png)

![Emotion mix 1](https://github.com/thanhtinhvan/Emotion-recognize/blob/master/Screenshoots/EmotionTest1.png)

![Emotion mix 2](https://github.com/thanhtinhvan/Emotion-recognize/blob/master/Screenshoots/EmotionTest2.png)


## Installation
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenFace](https://cmusatyalab.github.io/openface/)


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install scikit-learn==0.18.rc2
pip install python-tk
pip install Pillow
```
- Download pre-train models for OpenFace and Dlib. In [models] folder, run get-models.sh
- Download and extract pre-train model for Emotion which I was train at [link](https://drive.google.com/file/d/14uwMJnHLrpBB9WlQTMsINXRBu9XwX-v1/view?usp=sharing). Extract as [datas] folder inside [Emotions] folder.
## Usage
In Emotion folder, open terminal then:
```bash
python main.py
```

## Contributing
Pull requests are welcome. For major changes or can not run, please open an issue first to discuss what you would like to change or your issue. 
Notes: I will upload my src for lastest version soon.
