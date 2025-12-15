from setuptools import setup, find_packages

with open("README.MD", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aurora-xt-avsr",
    version="1.0.0",
    author="AVSR Team",
    author_email="anhvan170304@gmail.com",
    description="AURORA-XT: Audio-Visual Speech Recognition for Vietnamese",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnhVan17/Audio-Visual-Speech-Recognition",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "modal": [
            "modal>=0.56.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aurora-train=scripts.training:main",
            "aurora-preprocess=src.data.preprocessing:main",
        ],
    },
)
