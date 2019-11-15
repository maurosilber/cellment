from setuptools import setup

setup(
    name='cellment',
    version='0.1.0',
    packages=['cellment'],
    url='https://github.com/maurosilber/cellment',
    license='MIT',
    author='Mauro Silberberg',
    author_email='maurosilber@gmail.com',
    description='Segmentation of cells from fluorescence microscopy.',
    install_requires=['numpy', 'scipy', 'networkx', 'scikit-image', 'more_itertools']
)
