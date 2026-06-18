from setuptools import setup, find_packages

setup(
    name="battle_city_rl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.7.0",
        "pygame>=2.5.0",
        "moviepy>=1.0.3",
    ],
)
