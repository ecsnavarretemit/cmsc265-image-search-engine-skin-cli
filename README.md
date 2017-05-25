# CMSC 265 Exercise 8 - Image Search Engine (Skin)

## Requirements

1. Python 2.7.x
2. OpenCV 3.x

_Note: Make sure that Python is compiled with framework enabled when installing on macOS systems._

## Installing dependencies

This project requires a working installation of [OpenCV 3](http://opencv.org/). Please install this first before installing
the project dependencies.

Dependencies of this project can be installed via [PIP](https://pypi.python.org/pypi/pip).

Follow these steps to install the setup the application:

1. Run the command `virtualenv venv` to setup a Python 2 virtual environment.
2. Activate the virtual environment by running `source venv/bin/activate`
3. Install the project dependencies by running `pip install -r requirements.txt`
4. Locate your OpenCV Python bindings and type the command `echo <Python OpenCV bindings path> >> ./venv/lib/<python version>/site-packages/opencv3.pth`.

    Where:

    1. `<Python CV bindings path>`: is the site-packages folder inside the OpenCV installation.
    Please make sure you select the appropriate version of bindings that matches the Python version declared in the requirements.

    2. `<python version>`: is the version under your `venv` virtual environment folder

5. Create the folder `assets/img` inside the root of the project directory and place the input images inside the newly created folder.

## Running the Program

You can run the program by invoking `python run.py ` or `./run.py` to execute the image processing script. This will take time depending on the number of images to be processed so please be patient.

## License

MIT


