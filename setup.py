from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fake-news-detection",
    version="1.0.0",
    author="Your Team Name",
    author_email="your.email@example.com",
    description="A machine learning project for detecting fake news using NLP techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aashishlalwani/Fakenewsnew",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=4.0",
            "jupyter>=1.0",
        ],
        "web": [
            "streamlit>=1.8.0",
            "plotly>=5.5.0",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)