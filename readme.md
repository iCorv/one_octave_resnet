# Simple frame-wise Piano Transcription using Deep Convolutional Neural Networks
This project investigates the usage of ResNet for frame-wise piano transcription. The project is still work in progress, 
momentarily only notes from one octave of the piano are used to train and evaluate the model.
### Adding TF models to interpreter path
1. Get TF models from GitHub (https://github.com/tensorflow/models)
2. Locate the models folder in your PyCharm project folder
3. Add model to interpreter path in PyCharm: PyCharm -> Preferences -> Project Interpreter -> Show All -> Show paths for the selected interpreter -> Add  ../models/official
4. Mark the models folder as a source folder in project structure: PyCharm -> Preferences -> Project Structure -> right click models folder -> Choose Sources
5. Restart PyCharm