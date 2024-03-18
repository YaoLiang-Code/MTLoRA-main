import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xtuninglib",
    version="0.1.0",
    author="Yao Liang, Yuwei Wang, Yi Zeng",
    author_email="liangyao2023@ia.ac.cn",
    description="PyTorch implementation of Matrix-Transformation Based Low-Rank Adaptation (MTLoRA), A Brain-Inspired Method for Parameter-Efficient Fine-Tuning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YaoLiang-Code/MTLoRA-main",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)