import streamlit as st

import torch
from torchvision import transforms
from PIL import Image
import timm
import torch.nn as nn

#Load trained model
device = torch.device("cuda" if torch.cuda.is_available() 
                      else "cpu")
num_defects = 9

model = timm.create_model('vit_base_patch16_224',pretrained = True)
model.head = nn.Linear(model.head.in_features, num_defects )
model = model.to(device)
state_dict = torch.load("C:/Users/91994/FabricDefectDetection/src/vit_fabric_defect.pth",map_location = device)
model.load_state_dict(state_dict)

model.eval()
defects = ["Broken stitch","defect free", "hole", "horizontal", "lines", "Needle mark", "Pinched fabric", "stain", "Vertical"]

ImageSize = 224
transform = transforms.Compose([
    transforms.Resize((ImageSize,ImageSize)),
    transforms.ToTensor()
]
)

st.title("Fabric Defect Detection using Vision Transformer")
st.write("Upload a Fabric Image:")

Image_to_Analyze = st.file_uploader("Choose an image",type=["jpg","png","jpeg"])

if Image_to_Analyze is not None:
    fabric_image = Image.open(Image_to_Analyze).convert("RGB")
    st.image(fabric_image,caption="Uploaded Fabric",use_column_width = True)

    #Preprocess the image
    image = transform(fabric_image).unsqueeze(0).to(device)

    #prediction
    with torch.no_grad():
        outputs = model(image)
        _,predicted_id = outputs.max(1)
        pred_class = defects[predicted_id.item()]
    
    if pred_class == "defect free":
        st.sucess("The fabric is defect free")
    else:
        st.success(f"The fabric has defect. ")

