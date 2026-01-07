# Peatland-Fire-UNet-Project
Physics industry project with Airbus to design a U-Net that can segment and predict Peatland fires.

**Statement of Code Development**:
This processing and model pipeline was developed using Python 3.9. The deep learning architec-
ture is based on the U-Net Convolutional Neural Network (Ronneberger et al., 2015), specifically
adapted from an open-source reference (Tran, 2020) with PyTorch The custom ComboLoss func-
tion utilises standard Dice and Focal loss implementations referenced from the PyTorch commu-
nity. The geospatial data processing uses standard tiling and reprojection methods as outlined
in the official rasterio library documentation. Generative AI tools were used to assist with
code documentation, formatting for readability, and the implementation of terminal outputs to
monitor code progress.
