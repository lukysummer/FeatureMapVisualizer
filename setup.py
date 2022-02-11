import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name = "FeatureMapVisualizer",
    version="1.0.0",
    description = "CNN Feature Map Visualizations",
    long_description=README,
    long_description_content_type="text/markdown",
    url = "https://github.com/lukysummer/FeatureMapVisualizer",
    author = "Jahyun Shin",
    author_email = "lucrece.shin@mail.utorotno.ca",
    license = "MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
	"Intended Audience :: Developers",     
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
	"Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages = ["FeatureMapVisualizer"],
    include_package_data=True,
    install_requires = [
        "torch>=1.6",
    	"torchvision",
	"cv2"
    ],
    keywords=[
	"artificial intelligence",
	"machine learning",
        "visualization", 
        "image recognition", 
	"computer vision",
        "CNN"],
    download_url = "https://github.com/lukysummer/FeatureMapVisualizer/archive/refs/tags/v1.0.0.tar.gz"
)
