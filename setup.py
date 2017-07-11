from setuptools import setup

setup(name='dft',
      version='0.1',
      description='Density Functional Theory in Python',
      author='Gabriel Kabbe',
      author_email='gabriel.kabbe@chemie.uni-halle.de',
      license='MIT',
      packages=['dft'],
      install_requires=['numpy'],
      entry_points={
                    'console_scripts': [],
      },
      zip_safe=False)
