# Samogonka: Knowledge Distillation Framework

Samogonka is a Python framework designed to facilitate knowledge distillation experiments. Knowledge distillation is a technique where a smaller, less complex model (student) is trained to replicate the behavior of a larger, more complex model (teacher). This framework provides tools to train both teacher and student models using the CIFAR-10 dataset, and to perform knowledge distillation from the teacher to the student.

## Features

- CIFAR-10 data module for easy data loading and preprocessing.
- Predefined ResNet18 and ResNet50 models for student and teacher roles.
- Classification and distillation modules for training and knowledge transfer.
- Utility scripts for training models and running distillation processes.
- Integration with Weights & Biases for experiment tracking.

## Installation

To install Samogonka, you need to have Python 3.7 or higher. It is recommended to use a virtual environment. To install the package and its dependencies, follow these steps:

```bash
git clone https://github.com/your-username/samogonka.git
cd samogonka
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training the Teacher Model

To train the teacher model from scratch, run the following script:

```bash
python scripts/train_teacher_from_scratch.py
```

This will train a ResNet50 model on the CIFAR-10 dataset and save the best checkpoints.

### Training the Student Model

To train the student model from scratch, without knowledge distillation, run:

```bash
python scripts/train_student_from_scratch.py
```

This will train a ResNet18 model on the CIFAR-10 dataset and save the best checkpoints.

### Knowledge Distillation

To perform knowledge distillation from the teacher to the student, run:

```bash
python scripts/run_distillation_process.py
```

This script will load the pretrained teacher model, initialize the student model, and perform knowledge distillation.

## Project Structure

- `.gitignore`: Specifies intentionally untracked files to ignore.
- `LICENSE`: The MIT License file.
- `README.md`: The file you are currently reading.
- `environment.yml`: Conda environment file.
- `notebooks/simple_distillation.ipynb`: Jupyter notebook with a simple example of knowledge distillation.
- `requirements.txt`: List of Python package dependencies.
- `samogonka/`: Main package directory.
  - `__init__.py`: Initializes the package.
  - `datamodules/`: Data loading and preprocessing modules.
  - `models/`: Definitions of neural network models.
  - `modules/`: PyTorch Lightning modules for training and distillation.
  - `utils/`: Utility functions and classes.
- `scripts/`: Scripts for training and distillation processes.
- `setup.py`: Setup script for installing the package.

## License

Samogonka is MIT licensed, as found in the `LICENSE` file.
