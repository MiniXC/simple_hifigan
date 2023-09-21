from setuptools import setup, find_packages, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simple_hifigan",
    version="0.1.0",
    author="Christoph Minixhofer",
    author_email="christoph.minixhofer@gmail.com",
    description="A simple way to use HiFi-GAN.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MiniXC/simple_hifigan",
    packages=find_namespace_packages(),
    package_dir={"": "simple_hifigan"},
    package_data={
        "simple_hifigan.data": [
            "*.pth.tar",
            "*.json",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
