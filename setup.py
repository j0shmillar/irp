try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='aod-ds',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      author='edsml-jm4622')
