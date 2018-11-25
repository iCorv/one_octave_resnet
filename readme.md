# Simple frame-wise Piano Transcription using Deep Convolutional Neural Networks
This project investigates the usage of ResNet for frame-wise piano transcription. The project is still work in progress, 
momentarily only notes from one octave of the piano are used to train and evaluate the model.
### Adding TF models to interpreter path
1. Get TF models from GitHub (https://github.com/tensorflow/models)
2. Locate the models folder in your PyCharm project folder
3. Add model to interpreter path in PyCharm: PyCharm -> Preferences -> Project Interpreter -> Show All -> Show paths for the selected interpreter -> Add  ../models/official
4. Mark the models folder as a source folder in project structure: PyCharm -> Preferences -> Project Structure -> right click models folder -> Choose Sources
5. Restart PyCharm

### Running the Project on AWS Deep Learning AMI
1. Follow instructions in this blog: https://aws.amazon.com/de/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/
2. Run as python3 with tensorflow environment
    ```
    $ source activate tensorflow_p36
    ```
3. git clone this project
4. git clone the TF models (https://github.com/tensorflow/models)
    1. Add the top-level ***/models*** folder to the Python path with the command:
        ```
        $ export PYTHONPATH="$PYTHONPATH:/path/to/models"
        ```
    2. Install dependencies:
        ```
        $ pip3 install --user -r official/requirements.txt
        ```
5. Copy Dataset to your Instance with secure copy:
    ```
    $ scp -i /path/to/your/pem /the/file/on/your/system/dataset.npz ubuntu@ec2-18-188-163-228.us-east-2.compute.amazonaws.com:~/the/destination/folder/
    ```
6. Configure the model and hit
    ```
    $ python3 pop_resnet_mainpy
    ```
7. Copy trained model to your computer
    ```
    $ scp -i /path/to/your/pem -r ubuntu@ec2-18-224-61-22.us-east-2.compute.amazonaws.com:~/remote/model ./local/model/
    ```
### Running the Project on Windows 10 with Anaconda
1. Install Anaconda environment
2. Set up a tensorflow GPU environment. As easy as this: https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc
3. Get the TF models, as described above and link them to Python following these instructions: https://stackoverflow.com/questions/37006114/anaconda-permanently-include-external-packages-like-in-pythonpath
