from src.data_generator import generate_environment
from src.model import LinearModel
from src.regulator import SelfRegulator

x, y = generate_environment()
model = LinearModel(x.shape[1])
regulator = SelfRegulator()

for epoch in range(50):
    loss = model.update(x, y, regulator.lr)
    lr = regulator.adjust(loss)
    print(f"Epoch {epoch}: Loss={loss:.4f}, LR={lr:.6f}")
