# Self-Correction-Human-Parsing-CPU

This project is a CPU-compatible fork of the [Self-Correction-Human-Parsing](https://github.com/PeikeLi/Self-Correction-Human-Parsing) repository, tailored to run human parsing models on macOS or other CPU-based systems. The model generates segmentation masks for human body parts and clothing items, which can be used to extract specific elements, such as clothing, from input images.

## Prerequisites

Before starting, ensure you have the following:

- **Conda**: For environment management. Download from [Anaconda](https://www.anaconda.com/products/distribution).
- **Git**: To clone the repository. Install from [Git](https://git-scm.com/downloads).
- **Hardware**: A macOS machine (or any CPU-based system) with at least 8GB of RAM recommended.

## Setup for Harry, Neal, Praf, Pri

### 1. Clone the Repository

Clone the CPU version of the repository:

```
bash
git clone https://github.com/prafull2001/STC-img-extractor.git

cd STC-img-extractor

conda env create -f environment.yaml
conda activate schp
conda install -c conda-forge opencv
pip install -r requirements.txt

```
## Running the Model
### 1. Prepare Input Images
Place your input images (.jpg or .png) in a drectory you need to create called **input** directory. Use images of people with visible clothing for best results.

###2. Download the Pretrained Model
Download the pretrained LIP model from this link and place it in a directory you nee to create called **checkpoints**. Rename it to final.pth.
https://drive.usercontent.google.com/download?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH&authuser=0

###3. Generate Segmentation Masks
Run the model to create segmentation masks:

```
python schp_utils/simple_extractor.py --dataset lip --model-restore checkpoints/final.pth --input-dir input --output-dir output
```

## Extract clothes
After generating masks run:
```
python extract_clothing.py
```
