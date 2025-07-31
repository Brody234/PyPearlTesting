from pypearl import ArrayD2, ArrayD1, ArrayI2, Layer, ReLU, Softmax, CCE, SGD
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os, subprocess
os.environ["CXX"] = "/opt/homebrew/opt/llvm/bin/clang++"
os.environ["CC"] = "/opt/homebrew/opt/llvm/bin/clang"
os.environ["CXXFLAGS"] = "-isysroot " + subprocess.check_output(["xcrun","--show-sdk-path"], text=True).strip() + " -stdlib=libc++"

# USE PYPEARL VERSION 0.4.7, support on other packages not guaranteed

x = ArrayD2(3, 2)
for i in range(3):
    for j in range(2):
        x[i, j] = (i*j+3+j+i)/11

y = ArrayI2(3, 4)
y[0, 3] = 1
y[1, 1] = 1
y[2, 0] = 1

l1 = Layer(2, 100)

r1 = ReLU()

l2 = Layer(100, 100)

r2 = ReLU()

l3 = Layer(100, 100)

r3 = ReLU()

l4 = Layer(100, 4)

s = Softmax()

c = CCE()

o = SGD()

epochs = 1000
t0 = time.perf_counter()
for i in range(epochs):
    v1 = l1.forward(x)
    v2 = r1.forward(v1)
    v3 = l2.forward(v2)
    v4 = r2.forward(v3)
    v5 = l3.forward(v4)
    v6 = r3.forward(v5)
    v7 = l4.forward(v6)
    v8 = s.forward(v7)

    loss = c.forward(v8, y)

    if i == 0:
        initloss = loss
    dval = c.backward(v8, y)
    dval2 = l4.backward(dval)
    dval3 = r3.backward(dval2)
    dval4 = l3.backward(dval3)
    dval5 = r2.backward(dval4)
    dval6 = l2.backward(dval5)
    dval7 = r1.backward(dval6)
    l1.backward(dval7)
    o.optimize(l1)
    o.optimize(l2)
    o.optimize(l3)
    o.optimize(l4)
t1 = time.perf_counter()
elapsed = t1 - t0

print("\ntraining finished")
print(f"initial loss : {initloss:.6f}")
print(f"final loss   : {loss:.6f}")
print(f"elapsed time : {elapsed:.2f} s "
      f"(≈ {elapsed/epochs*1e6:.1f} µs / epoch)")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.empty(3, 2, dtype=torch.float32).to(device)
for i in range(3):
    for j in range(2):
        x[i, j] = (i * j + 3 + j + i) / 11

y = torch.tensor([3, 1, 0], dtype=torch.long, device=device)

model = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 4)
).to(device)



criterion  = nn.CrossEntropyLoss()
optimizer  = optim.SGD(model.parameters(), lr=1e-3)
if torch.__version__ >= "2.0":
    backend = "aot_eager" if device.type == "mps" else "inductor"
    model   = torch.compile(model, backend=backend, mode="max-autotune")


epochs = 1000

t0 = time.perf_counter()
init_loss = None

for epoch in range(epochs):
    logits = model(x)
    loss   = criterion(logits, y)

    if epoch == 0:
        init_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

t1 = time.perf_counter()
elapsed = t1 - t0

print("\ntraining finished")
print(f"initial loss : {init_loss:.6f}")
print(f"final loss   : {loss.item():.6f}")
print(f"elapsed time : {elapsed:.2f} s "
      f"(≈ {elapsed / epochs * 1e6:.1f} µs / epoch)")

probs = torch.softmax(model(x), dim=1)
print("\nsoftmax outputs:")
print(probs)

