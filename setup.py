from setuptools import setup, find_packages, setuptools
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
setup(
    name='RoboCam',
    version='1.0.0.35',
    description='Python library for RoboCam',
    author='Roborobo',
    author_email='roborobolab@gmail.com',
    url = 'https://eng.roborobo.co.kr/main',
    download_url = 'https://github.com/RoboroboLab/RoboCam/archive/master.tar.gz',
    license='MIT',
    packages = setuptools.find_packages(),
    keywords = ['RoboCam','roborobo'],
    python_requires='>=3',
    long_description = long_description,
    long_description_content_type='text/markdown',
    package_data =  {
        'RoboCam' : [
            'res/model/face_detector.tflite',
            'res/model/face_keypoints.tflite',
            'res/model/face_recognizer.tflite',
            'res/model/iris_landmark.tflite',
            'res/model/mnist_model.tflite'
    ]},
    zip_safe=False,
    install_requires=[
        'opencv-contrib-python==4.7.0.72',
        'tflite==2.10.0',
        'tensorflow==2.20.0',
        'numpy==1.26.1'
    ], 
    classifiers = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10', 
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12', 
    ]
)
