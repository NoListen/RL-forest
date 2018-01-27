from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='RL_forest',
      packages=[package for package in find_packages()
                if package.startswith('baselines')],
      install_requires=[
          'gym[mujoco,atari,classic_control]',
          'scipy',
          'tqdm',
          'joblib',
          'zmq',
          'dill',
          'azure==1.0.3',
          'progressbar2',
      ],
      description="RL_forest: implement variants of reinforcement learning",
      author="Lisheng Wu",
      url='https://github.com/nolisten/RL_forest',
      version="0.0.1")

