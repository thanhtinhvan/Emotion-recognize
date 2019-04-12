# Emotion recognize

This is my thesis project at University. Using OpenFace to detect face then training a Deep Learning model CNNs to recognize emotion on the faces.
Best compatible with Python 2.7 on Ubuntu 16.04.

## Installation

- [OpenFace](https://cmusatyalab.github.io/openface/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

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