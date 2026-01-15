import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

# 1. Load Pre-trained Model (Weights are downloaded automatically)
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()  # Set to evaluation mode

# 2. Define the Image Transformation (Preprocessing)
# This matches the training data format of ImageNet
preprocess = weights.transforms()

# 3. The Recognition Function
def recognize_image(image_path):
    try:
        # Load and convert image to RGB
        img = Image.open(image_path).convert('RGB')
        
        # Preprocess the image and add a batch dimension
        batch = preprocess(img).unsqueeze(0)

        # Perform Inference
        with torch.no_grad():
            prediction = model(batch).squeeze(0)
            confidences = torch.nn.functional.softmax(prediction, dim=0)
        
        # Get the top result
        class_id = prediction.argmax().item()
        score = confidences[class_id].item()
        category_name = weights.meta["categories"][class_id]

        return {"label": category_name, "confidence": f"{score:.2%}"}
    except Exception as e:
        return {"error": str(e)}
# This part actually runs when you execute the script
if __name__ == "__main__":
    result = recognize_image("test.jpeg")
    print("\n--- AI RECOGNITION RESULT ---")
    print(result)
    print("-----------------------------\n")
# Quick test (Uncomment below if you have a 'test.jpg' in your folder)
# print(recognize_image("test.jpg"))