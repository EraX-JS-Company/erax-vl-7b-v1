from setuptools import setup, find_packages

setup(
    name="erax-vl-7b-v1",
    version="0.1.0",
    description="EraX-VL-7B-V1 - A multimodal vision-language model based on Qwen2-VL-7B architecture.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EraX-JS-Company/erax-vl-7b-v1",
    author="EraX",
    author_email="nguyen@erax.ai",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=[
        "json_repair",
        "numpy",
        "matplotlib",
        "tqdm",
        "pillow",
        "PyMuPDF",
        "opencv-python",
        "python-dotenv",
        "requests",
        "regex"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.6",
)
