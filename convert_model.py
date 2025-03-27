import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from torchvision import transforms
from inplace_abn import ABN  # For weight adjustment

# Import ResNet and Bottleneck from a_ce2p.py
from a_ce2p import ResNet, Bottleneck

# Define the model path
model_path = '/Users/prafullsharma/Desktop/STC-img-extractor/checkpoints/final.pth'

# Load the model
model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=20)  # ResNet-101 with 20 classes
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Remove 'module.' prefix if present (from DataParallel training)
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

# Adjust ABN weights to ensure positive scaling factors
for module in model.modules():
    if isinstance(module, ABN):
        module.weight.data = module.weight.data.abs() + 1e-5

model.eval()

# Define a wrapper to select fusion_result and upsample
class ParsingWrapper(nn.Module):
    def __init__(self, model):
        super(ParsingWrapper, self).__init__()
        self.model = model
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = x / 255.0  # Scale input from [0, 255] to [0, 1]
        x = self.normalize(x)  # Apply ImageNet normalization
        outputs = self.model(x)
        fusion_result = outputs[0][1]  # Select fusion_result from [[parsing, fusion], [edge]]
        # Upsample to input size (473, 473)
        fusion_result = F.interpolate(fusion_result, size=(473, 473), mode='bilinear', align_corners=True)
        return fusion_result

# Verify output shape
input_tensor = torch.rand(1, 3, 473, 473)
wrapped_model = ParsingWrapper(model)
with torch.no_grad():
    output = wrapped_model(input_tensor)
print("Output shape:", output.shape)  # Should be [1, 20, 473, 473]

# If the shape is correct, proceed with conversion
if output.shape == (1, 20, 473, 473):
    # Set wrapped model to eval mode before tracing
    wrapped_model.eval()

    # Trace the model for CoreML conversion
    example_input = torch.rand(1, 3, 473, 473) * 255  # Input in [0, 255] range
    traced_model = torch.jit.trace(wrapped_model, example_input)

    # Convert to CoreML with neural network format
    mlmodel = ct.convert(
        traced_model,
        convert_to="neuralnetwork",  # Ensure neural network format for NeuralNetworkBuilder
        inputs=[ct.ImageType(name="input", shape=example_input.shape, scale=1.0, bias=[0, 0, 0])],
        outputs=[ct.TensorType(name="logits")],
    )

    # Add argmax layer for segmentation labels
    spec = mlmodel.get_spec()
    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
    builder.add_argmax(name="argmax", input_name="logits", output_name="labels", axis=1)
    new_mlmodel = ct.models.MLModel(builder.spec)
    new_mlmodel.save("HumanParsingModel.mlmodel")
    print("Model successfully converted and saved as HumanParsingModel.mlmodel")
else:
    print("Model output shape mismatch. Please verify the model architecture and input size.")