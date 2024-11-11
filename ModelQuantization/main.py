import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import time

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]
output_dim = len(set(y_train))
model = LogisticRegressionModel(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

quantized_model = quantize_model(model)

quantized_model.eval()
with torch.no_grad():
    start_time = time.time()
    quantized_outputs = quantized_model(X_test_tensor)
    _, quantized_predicted = torch.max(quantized_outputs, 1)
    quantized_inference_time = time.time() - start_time

quantized_accuracy = (quantized_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
print(f"Quantized PyTorch Model Accuracy: {quantized_accuracy:.4f}")
print(f"Quantized PyTorch Model Inference Time: {quantized_inference_time:.6f} seconds")

sklearn_model = LogisticRegression(max_iter=10000)
sklearn_model.fit(X_train, y_train)

sklearn_accuracy = sklearn_model.score(X_test, y_test)
print(f"scikit-learn Logistic Regression Model Accuracy: {sklearn_accuracy:.4f}")

start_time = time.time()
sklearn_model.predict(X_test)
sklearn_inference_time = time.time() - start_time
print(f"scikit-learn Logistic Regression Inference Time: {sklearn_inference_time:.6f} seconds")

time_saved_vs_sklearn = sklearn_inference_time - quantized_inference_time
print(f"\nTime Saved by Quantized PyTorch Model vs scikit-learn Model: {time_saved_vs_sklearn:.6f} seconds")

print("\nComparison Summary:")
print(f"Accuracy (Quantized PyTorch): {quantized_accuracy:.4f}")
print(f"Accuracy (scikit-learn): {sklearn_accuracy:.4f}")

print(f"Inference Time (Quantized PyTorch): {quantized_inference_time:.6f} seconds")
print(f"Inference Time (scikit-learn): {sklearn_inference_time:.6f} seconds")

print(f"\nTime Saved by Quantized PyTorch Model vs scikit-learn Model: {time_saved_vs_sklearn:.6f} seconds")
