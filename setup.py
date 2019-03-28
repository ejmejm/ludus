import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ludus",
    version="0.1a3",
    author="Edan Meyer",
    author_email="ejmejm98@gmail.com",
    description="Reinforcement learning library to expediate application and research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ejmejm/ludus",
    packages=["ludus"],
    setup_requires=["numpy"],
    install_requires=[
        "gym",
        "opencv-python"
    ],
    extras_require={
        "tf": ["tensorflow>=1.10"],
        "tf-gpu": ["tensorflow-gpu>=1.10"],
        "atari": ["gym[atari]"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
