from pypearl import ArrayD2, ArrayD1, ArrayI2, Layer, ReLU, Softmax, CCE, SGD
import time
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os, subprocess
import numpy as np
os.environ["CXX"] = "/opt/homebrew/opt/llvm/bin/clang++"
os.environ["CC"] = "/opt/homebrew/opt/llvm/bin/clang"
os.environ["CXXFLAGS"] = "-isysroot " + subprocess.check_output(["xcrun","--show-sdk-path"], text=True).strip() + " -stdlib=libc++"
def pypearl_init(m):
    if isinstance(m, nn.Linear):
        fan_in, fan_out = m.weight.size()   # (out, in) → (fan_out, fan_in)
        # nn.init calculates fan_in/fan_out the other way round,
        # so we just reproduce your formula explicitly
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        with torch.no_grad():
            nn.init.uniform_(m.weight, -0.5 * limit, 0.5 * limit)  # extra ×0.5
            nn.init.constant_(m.bias, 0.0)                        # biases = 0


epochCONST = 100
# USE PYPEARL VERSION 0.4.7, support on other packages not guaranteed

x = ArrayD2(3, 2)
for i in range(3):
    for j in range(2):
        x[i, j] = (i*j+3+j+i)/11

y = ArrayI2(3, 4)
y[0, 3] = 1
y[1, 1] = 1
y[2, 0] = 1

l1 = Layer(2, 1000)

r1 = ReLU()

l2 = Layer(1000, 1000)

r2 = ReLU()

l3 = Layer(1000, 1000)

r3 = ReLU()

l4 = Layer(1000, 4)

s = Softmax()

c = CCE()

o = SGD()

epochs = epochCONST
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
    #dvaltest = np.zeros((3, 4))
    #for i in range (3):
    #    for j in range (4):
    #        dvaltest[i, j] = dval[i, j]
    #print(np.linalg.norm(dvaltest))  # pypearl*/
    dval1 = s.backward(dval)
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
    nn.Linear(2, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 4)
).to(device)

model.apply(pypearl_init)


criterion  = nn.CrossEntropyLoss()
optimizer  = optim.SGD(model.parameters(), lr=1e-3)
if torch.__version__ >= "2.0":
    backend = "aot_eager" if device.type == "mps" else "inductor"
    model   = torch.compile(model, backend=backend, mode="max-autotune")


epochs = epochCONST

t0 = time.perf_counter()
init_loss = None

for epoch in range(epochs):
    logits = model(x)

    loss   = criterion(logits, y)
    logits.retain_grad()


    if epoch == 0:
        init_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(torch.norm(logits.grad))  # torch


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

