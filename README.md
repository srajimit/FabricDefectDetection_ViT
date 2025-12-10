# FabricDefectDetection_ViT
To identify whether Fabric Has defect using Vision Transformer.

The dataset taken includes the classes which are specified as:
Broken stitch:112
Needle mark:108
Pinched fabric:108
Vertical:101
defect free:1666
hole:281
horizontal:136
lines:157
stain:398

The Pretrained Vision Transformer is loaded and the model is trained for 10 epochs. The model produced Training Loss: 0.5792, Training Accuracy: 82.51%, Validation Loss: 0.9225,and Validation Accuracy: 58.96%. Because of resource constraints, the experiments is conducted for 10 epochs.

TextileDefectDetection_ViT.ipynb # This ipynb file is developed in google colab. 
	Input: Dataset
	Output: vit_fabric_defect.pth

app.py 
	Loads the model. It asks for the input image and identify whether the image is having defect or not.

The application has to be executed as:
	python -m streamlit run app.py


**Download Datasets from Kaggle:** https://www.kaggle.com/datasets/ziya07/multi-class-fabric-defect-detection-dataset

