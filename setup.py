import setuptools

with open("requirements.txt", "rt") as requirements_file:
    requirements = list(filter(None, map(str.strip,
                                         requirements_file.readlines())))
    
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='easydashboard',
    packages=['easydashboard'],
    version='0.0.1',
    license='MIT',
    description='Testing installation of EasyDashboard Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Harshal Soni',
    author_email='arcadiahms@gmail.com',
    url='https://github.com/harshalsonioo1/mpr-asi',
    zip_safe=False,
    keywords=["pypi", "easydashboard", "tutorial"],
    install_requires=requirements,
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