import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print(f"Available GPUs: {available_gpus}")
else:
    print("No GPUs available")


np.random.seed(42)
X = np.random.rand(100, 1)
y = 3.5 * X.squeeze() + np.random.randn(100) * 0.5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

mlflow.autolog()

model = LinearRegression()
model.fit(X_train, y_train)

