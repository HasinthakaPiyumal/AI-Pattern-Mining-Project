import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import base64

app = FastAPI()

# Global variable to store the style image tensor
style_image_tensor = None

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
loader = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

unloader = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]), # Reverse normalization
    transforms.ToPILImage()
])

# Function to load and preprocess an image
def image_loader(image_bytes, imsize=512):
    image = Image.open(io.BytesIO(image_bytes))
    # Resize while maintaining aspect ratio
    if image.size[0] > image.size[1]:
        new_width = imsize
        new_height = int(imsize * image.size[1] / image.size[0])
    else:
        new_height = imsize
        new_width = int(imsize * image.size[0] / image.size[1])
    image = image.resize((new_width, new_height))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Content loss
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach() # Detach target from graph

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# Style loss
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into 
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d) # normalize the values by dividing by the number of elements in each feature map.


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Get VGG-19 model and extract layers
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_model_and_losses(cnn, style_img, content_img, content_layers, style_layers):
    cnn = cnn.to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Function to get input optimizer
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# Main style transfer function
def run_style_transfer(cnn, style_img, content_img, input_img, num_steps=300, style_weight=1000000, content_weight=1):

    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img, content_layers_default, style_layers_default)
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# Load pre-trained VGG model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

@app.post("/upload_style_image")
async def upload_style_image(file: UploadFile = File(...)):
    global style_image_tensor
    try:
        image_bytes = await file.read()
        style_image_tensor = image_loader(image_bytes)
        return {"message": "Style image uploaded successfully"}
    except Exception as e:
        return {"message": f"Failed to upload style image: {e}"}

@app.post("/apply_style_transfer")
async def apply_style_transfer(file: UploadFile = File(...)):
    global style_image_tensor
    if style_image_tensor is None:
        return {"message": "Please upload a style image first using /upload_style_image"}

    try:
        content_image_bytes = await file.read()
        content_image_tensor = image_loader(content_image_bytes)

        # Initialize the input image with a copy of the content image
        input_img = content_image_tensor.clone()

        output_tensor = run_style_transfer(
            cnn=cnn, style_img=style_image_tensor, content_img=content_image_tensor, input_img=input_img
        )

        output_image = unloader(output_tensor.cpu().squeeze(0))

        buffered = io.BytesIO()
        output_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {"stylized_image": img_str}

    except Exception as e:
        return {"message": f"Failed to apply style transfer: {e}"}
