from setuptools import setup


setup(
    name="baleian-sc2",
    packages=[
        "baleian.sc2",
    ],
    version="0.0.1",
    description="A StarCraft II AI",
    license="MIT",
    author="baleian",
    author_email="baleian@gmail.com",
    url="https://github.com/baleian/python-sc2",
    keywords=["StarCraft", "StarCraft 2", "StarCraft II", "AI", "Bot"],
    install_requires=[
        "pysc2",
        "numpy",
        "absl"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Games/Entertainment",
        "Topic :: Games/Entertainment :: Real Time Strategy",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
