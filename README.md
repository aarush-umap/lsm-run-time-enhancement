# Training denoising and super-resolution model for laser scanning microscopy.  
Self-supervised denoising and single-image super-resolution for denoising and super-resolution.  
Related project: [SmartPath](github.com/uw-loci/smart-wsi-scanner), [WSISR](github.com/uw-loci/demo_wsi_superres), [Pycro-Manager](https://github.com/micro-manager/pycro-manager)

## Installation
Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html).    
Install required packages:
```
    $ conda env create --name denoising --file env.yml
    $ conda activate denoising
```

# Training denoising model
### Prepare datasets
#### Datasets collected by SmartPath or Pycro-Manager
Put the Pycro-Manager stack datasets (individual dataset folders) into a directory named `raw-data/[RESOLUTION]/`.  
`[RESOLUTION]` = resolution of the images. 
Open `dataset-prepare.ipynb` in JupyterLab. Configrue the parameters and run the sections in the notebook.  The code will produce z slices of each location of the datasets and saved them in `data/[RESOLUTION]` folder.  
#### Image slicecs
Otherwise, you could put individual `.png` images inside `data/[RESOLUTION]` folder.  

### Training and testing
Open `train-denoising.ipynb`. Exceuting the training loop after adjusting parameters and viewing example images from the dataset. Once the training is done, use the testing loop to see results on the validation set. Output images will be printed in folder `print`. Model weights are saved in `model-weights`.  

# Training SISR model
### Prepare datasets
Put input `.png` images inside `data/SISR/input` folder and target `.png` images inside `data/SISR/Target` folder.  

### Training and testing
Open `train-SISR.ipynb`. Exceuting the training loop after adjusting parameters and viewing example images from the dataset. Once the training is done, use the testing loop to see results on the validation set. Output images will be printed in folder `print-sr`. Model weights are saved in `model-weights`.  

