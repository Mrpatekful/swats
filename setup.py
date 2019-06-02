import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pytorch-swats',
    version='0.1.0',
    author='Patrik Purgai',
    author_email='purgai.patrik@gmail.com',
    description='PyTorch implementation of SWATS algorithm.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/Mrpatekful/swats",
    packages=setuptools.find_packages(),
    install_requires=['torch'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
