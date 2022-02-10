from distutils.core import setup

setup(
    name = "FeatureMapVisaulizer",
    packages = ["FeatureMapVisaulizer"],
    version="1.0.0",
    license = "MIT",
    description = "Feature Visualizations for CNN-based image classification models to get insights about their predictions",
    author = "Jahyun Shin",
    author_email = "lucrece.shin@mail.utorotno.ca",
    url = "https://github.com/lukysummer/FeatureMapVisualizer",
    download_url = "https://github.com/lukysummer/FeatureVisualizer/archive/refs/tags/v1.0.0.tar.gz"
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
    python_requires=">=3.6",
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
)