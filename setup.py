import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ludus",
    version="0.1a1",
    author="Edan Meyer",
    author_email="ejmejm98@gmail.com",
    description="Reinforcement learning library to expediate application and research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ejmejm/ludus",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
