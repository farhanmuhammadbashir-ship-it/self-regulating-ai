from src.data_generator import generate_environment
from src.model import LinearModel
from src.regulator import SelfRegulator

x, y = generate_environment()
model = LinearModel(x.shape[1])
regulator = SelfRegulator()

print(f"{'Epoch':<10} | {'Loss':<15} | {'Learning Rate':<15}")
print("-" * 45)

for epoch in range(50):
    loss = model.update(x, y, regulator.lr)
    lr = regulator.adjust(loss)
    print(f"{epoch:<10} | {loss:<15.4f} | {lr:<15.6f}")
