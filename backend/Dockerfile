# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt and install dependencies
# We use the --no-cache-dir to keep the image size smaller
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend code and the trained model
COPY app_fgsm.py fgsm.py mnist_model.bin ./

# Set the CMD to your handler (file_name.handler_name)
# Since we added 'handler = Mangum(app)' in app_fgsm.py:
CMD ["app_fgsm.handler"]