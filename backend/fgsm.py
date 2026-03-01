import torch
import torch.nn.functional as F

class Attack:
    def __init__(self, model):
        """
        Initialize with the pretrained model to be attacked.
        """
        self.model = model
        self.model.eval()  # Ensure model is in evaluation mode

    def fgsm_attack(self, image, epsilon, data_grad):
        """
        Core FGSM logic: move the image pixels in the direction of the gradient.
        """
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        
        # Adding clipping to maintain the [0,1] range for image data
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image

    def run(self, image, target_label, epsilon):
        """
        Performs the full attack cycle: 
        1. Forward pass
        2. Calculate loss
        3. Backward pass to get gradients
        4. Apply FGSM perturbation
        """
        # Ensure the image requires gradient calculation
        image.requires_grad = True

        # Forward pass the data through the model
        output = self.model(image)
        init_pred = output.max(1, keepdim=True)[1] # Get the index of the max log-probability

        # If the initial prediction is already wrong, don't bother attacking
        if init_pred.item() != target_label.item():
            return None, init_pred.item(), init_pred.item()

        # Calculate the loss (typically CrossEntropy)
        loss = F.nll_loss(output, target_label)

        # Zero all existing gradients
        self.model.zero_grad()

        # Backward pass to calculate gradients
        loss.backward()

        # Collect the gradient of the input image
        data_grad = image.grad.data

        # Call FGSM Attack
        perturbed_data = self.fgsm_attack(image, epsilon, data_grad)

        # Re-classify the perturbed image
        output_perturbed = self.model(perturbed_data)
        final_pred = output_perturbed.max(1, keepdim=True)[1]

        return perturbed_data, init_pred.item(), final_pred.item()