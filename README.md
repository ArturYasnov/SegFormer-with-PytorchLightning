# SegFormer-with-PytorchLightning

This project contains an example of training the SegFormer model for image segmentation using Pytorch Lightning.

SegFormer is a simple and efficient semantic segmentation architecture which unifies Transformers with lightweight decoders.

The project includes scripts for training a model, a Jupyter notebook example of using the trained model for inference, 
as well as some sample results from the model after 20 epochs of training on [UAVid urban scenes dataset](https://www.kaggle.com/datasets/dasmehdixtr/uavid-v1). 

Project based on 
[NielsRogge SegFormer repo](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/README.md)
and [huggingface segformer documentation](https://huggingface.co/docs/transformers/model_doc/segformer).


Results of a model after first 20 epochs of training on 600 images dataset.

Image:

![Image](data/image1.png)

Prediction:

![Image](data/prediction1.png)

Mask:

![Image](data/mask1.png)
