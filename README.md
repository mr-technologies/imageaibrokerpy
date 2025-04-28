# `imageaibrokerpy`

`imageaibroker.py` application demonstrates how to use AI/ML detection algorithms together with Python API of [MRTech IFF SDK](https://mr-te.ch/iff-sdk).
It is located in `samples/08_imageai_py` directory of IFF SDK package.
Application comes with example configuration file (`imageaibroker.json`) providing the following functionality:

* acquisition from XIMEA camera
* color pre-processing on GPU:
  * black level subtraction
  * histogram calculation
  * white balance
  * demosaicing
  * color correction
  * gamma
  * image format conversion
* automatic control of exposure time and white balance
* image export to the user code

Additionally example code uses [ImageAI](https://github.com/OlafenwaMoses/ImageAI) Python library to detect objects and [OpenCV](https://opencv.org/) library to render images on the screen, both of which should be installed in the system.
