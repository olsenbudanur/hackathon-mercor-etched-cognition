from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="eeg_enhanced_lm",
    version="0.1.0",
    author="Contributors",
    author_email="your.email@example.com",
    description="EEG-Enhanced Language Model with Mixture-of-Experts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hackathon-mercor-etched-cognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "eeg-lm-demo=kevin_moe_demo.demo:main",
        ],
    },
)
