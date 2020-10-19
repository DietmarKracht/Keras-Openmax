import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-openmax",
    version="1.0.0",
    author="Clemens Brackmann",
    author_email="clemens.brackmann@web.de",
    description="Keras implementation of Openmax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DietmarKracht/Keras-Openmax",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)