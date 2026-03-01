from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64

# Import your Attack class from fgsm.py
from backend.fgsm import Attack

# 1. Define the EXACT same model architecture used in training
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

app = FastAPI()

# 2. Add CORS Middleware (Crucial for Next.js to communicate with API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Model and Attacker
device = torch.device('cpu')
model = Net()
try:
    model.load_state_dict(torch.load("mnist_model.bin", map_location=device))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Warning: mnist_model.bin not found. Run evaluation script first!")

model.eval()
attacker = Attack(model)

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.get("/")
async def health_check():
    return {"status": "running", "model": "MNIST-CNN"}

@app.post("/attack")
async def perform_attack(
    file: UploadFile = File(...),
    epsilon: float = Form(...)
):
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image_tensor = transform(image).unsqueeze(0) # Shape: [1, 1, 28, 28]
        
        # Get original prediction
        output = model(image_tensor)
        # Fix for the 500 error: ensure target is a 1D LongTensor
        target_label = output.max(1, keepdim=True)[1].view(-1).long()
        original_pred = target_label.item()

        # Run FGSM Attack
        perturbed_data, init_pred, final_pred = attacker.run(image_tensor, target_label, epsilon)

        # Convert adversarial tensor back to Base64 PNG for Frontend
        # perturbed_data is [1, 1, 28, 28], we need [28, 28]
        adv_img_np = perturbed_data.squeeze().detach().cpu().numpy()
        adv_img_pil = Image.fromarray((adv_img_np * 255).astype('uint8'))
        
        buffered = io.BytesIO()
        adv_img_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "original_prediction": int(original_pred),
            "adversarial_prediction": int(final_pred),
            "adversarial_image": f"data:image/png;base64,{img_base64}",
            "success": bool(original_pred != final_pred),
            "epsilon_used": epsilon
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

