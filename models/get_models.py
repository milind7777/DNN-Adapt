import torch
import timm
from efficientnet_pytorch import EfficientNet
import torchvision.models as models

def get_model(model, onnx_version, onnx_path):
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,                      # Model to export
        dummy_input,                # Input to the model
        onnx_path,                  # Output ONNX file
        export_params=True,         # Store the trained parameter weights inside the ONNX file
        opset_version=onnx_version,           # ONNX version to use
        do_constant_folding=True,   # Optimize by pre-computing constant expressions
        input_names=['input'],      # Name of the input node
        output_names=['output'],    # Name of the output node
        dynamic_axes={'input': {0: 'batch_size'},    # Allows for dynamic batch size
                    'output': {0: 'batch_size'}}
    )

    print(f"model exported to {onnx_path}")

# Load a pre-trained ResNet model (e.g., ResNet18)
model = models.resnet18(pretrained=True)
get_model(model, 10, "resnet18.onnx")

model = timm.create_model('vit_base_patch16_224', pretrained=True)
get_model(model, 14, "vit16.onnx")

model = EfficientNet.from_pretrained('efficientnet-b0')
get_model(model, 10, "efficientnetb0.onnx")