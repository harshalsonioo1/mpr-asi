import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='easydashboard',
    packages=['easydashboard'],
    version='0.0.1',
    license='MIT',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Harshal Soni',
    author_email='arcadiahms@gmail.com',
    url='https://github.com/Muls/toolbox_public',
    project_urls = {
        "Bug Tracker": "https://github.com/Muls/toolbox/issues"
    },
    install_requires=['requests'],

    download_url="https://github.com/mike-huls/toolbox_public/archive/refs/tags/0.0.3.tar.gz",
    keywords=["pypi", "easydashboard", "tutorial"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Documentation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ]
)