
# PyQt5-Neural Networks-Training-GUI-With-TensorFlow & Keras

A PyQt5-based GUI application for training and evaluating deep learning models on image classification tasks. This toolbox provides an intuitive interface for selecting pre-trained models, configuring training parameters, and visualizing results.


## Screenshots

### Main Interface
<p align="center">
<img src="https://github.com/user-attachments/assets/46314903-9e9b-4080-82c1-d69dcbf68f10" width="400">


*The main interface where you can configure your model and training parameters*

### Training Progress
<p align="center">
<img src="https://github.com/user-attachments/assets/79b685cd-4468-43fe-8c95-ca3477ebd6f7" width="300">


*Real-time training progress window showing metrics and progress bars*

## Features

- Support for multiple pre-trained models:
  - VGG16
  - ResNet50
  - MobileNetV2
  - Xception
  - DenseNet121
- Customizable training parameters
- Real-time training progress visualization
- Data augmentation support
- Training history plots
- Confusion matrix visualization
- Dark mode UI

## Requirements

- Python 3.7+
- TensorFlow 2.x
- PyQt5
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/emirsahan00/PyQt5-Neural-Networks-Training-GUI-With-TensorFlow-Keras.git
```

2. Install the required packages:
```bash
pip install tensorflow numpy PyQt5 matplotlib seaborn scikit-learn
```

## Project Structure

```
deep-learning-toolbox/
├── config.py           # Configuration and styling
├── data_generator.py   # Data augmentation and generation
├── model_builder.py    # Model architecture and optimization
├── callbacks.py        # Custom training callbacks
├── visualization.py    # Training visualization utilities
├── gui.py             # GUI implementation
└── main.py            # Application entry point
```

## Usage

1. Run the application:
```bash
python main.py
```

2. In the GUI:
   - Select your dataset directory (should have 'train' and 'test' subdirectories)
   - Choose a pre-trained model
   - Configure training parameters
   - Start training
   - Monitor progress in real-time
   - View training history and confusion matrix plots

## Dataset Structure

Your dataset should be organized as follows:
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
└── test/
    ├── class1/
    │   └── image5.jpg
    └── class2/
        └── image6.jpg
```

## Training Configuration

- **Model Selection**: Choose from VGG16, ResNet50, MobileNetV2, Xception, or DenseNet121
- **Optimizer**: SGD, Adam, RMSprop, or Nadam
- **Learning Rate**: Configurable through the GUI
- **Batch Size**: Adjustable for both training and validation
- **Number of Epochs**: Set through the GUI


## License

This project is licensed under the MIT License
