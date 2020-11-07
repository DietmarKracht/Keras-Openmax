import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-openmax",
    version="1.0.2",
    author="Clemens Brackmann",
    author_email="clemens.brackmann@web.de",
    description="Keras implementation of Openmax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DietmarKracht/Keras-Openmax",
    py_modules=["Openmax", "compute_openmax", "EVT_fitting", "openmax_utils"],
    install_requires=[
        'Cython>=0.29.15',
        'Keras==2.3.1',
        'libmr>=0.1.9',
        'matplotlib>=3.2.0',
        'numpy>=1.18.1',
        'Pillow>=7.0.0',
        'scipy==1.4.1',
        'tensorflow==2.1.0',
        'tqdm>=4.43.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)