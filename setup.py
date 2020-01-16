from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym_pcgrl',
      version='0.4.0',
      install_requires=['gym', 'numpy>=1.17', 'pillow'],
      author="Ahmed Khalifa",
      author_email="ahmed@akhalifa.com",
      description="A package for \"Procedural Content Generation via Reinforcement Learning\" OpenAI Gym interface.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/amidos2006/gym-pcgrl",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ]
)
