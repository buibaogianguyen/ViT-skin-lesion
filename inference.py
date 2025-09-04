import torch
import os
from PIL import Image
from torchvision import transforms
from model.vit import VisionTransformer
from database_storing import store_to_db

def infer(model, img_path, transform, device):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    
    model.eval()

    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    return probs

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VisionTransformer(img_shape=224, patch_size=16, depth=12, hidden_dim=768, num_heads=12, mlp_dim=3072,num_classes=9)

    try:
        model.load_state_dict(torch.load('vit.pth', map_location=device))
    except FileNotFoundError:
        raise RuntimeError("Model file vit_model.pth was not found. Run model/train.py first.")
    
    model.to(device)
    
    image_path = ""
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    try:
        probs = infer(model, image_path, transform, device)
        label_map = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        print("Probabilities:")
        for i, prob in enumerate(probs):
            print(f"{label_map[i]}: {prob:.4f}")
            
        store_to_db(image_id, probs)

    except Exception as e:
        print(f"Inference error: {e}")
