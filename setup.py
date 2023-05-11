from setuptools import setup, find_packages

setup(name='spliformer',
      description='Spliformer',
      version='1.0',
      author='TJ-Zhanglab',
      author_email='tjzhanglab@163.com',
      license='GPLv3',
      url='https://github.com/TJ-zhanglab/Spliformer,
      packages=['spliformer'],
      install_requires=['pyfaidx>=0.6.3.1',
                        'pysam>=0.19.1',
                        'numpy>=1.22.4',
                        'pandas>=1.2.3',
                       'seaborn>=0.11.2'],
#                        'torch>=1.5.0'],
      extras_require={'cpu': ['torch>=1.5.0'],
                      'gpu': ['torch>=1.9.0']},
      package_data={'spliformer':['weights/*']},
#       py_modules=["model/Network_general", "model/Network_motif","model/spliformer_motif","model/spliformer_general"],
      entry_points={'console_scripts': ['spliformer=spliformer.main:main']} 
     )
      #test_suite='tests')
