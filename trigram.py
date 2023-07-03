import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Read in names
words = open('names.txt', 'r').read().splitlines()

# All possible characters
chars = ['.'] + sorted(list(set(''.join(words))))

N = torch.zeros((27, 27, 27), dtype=torch.int32)

chars = ['.'] + sorted(list(set(''.join(words))))
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for c, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.'] # Speical start token
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        N[stoi[ch1], stoi[ch2], stoi[ch3]] += 1

print(N.shape)
N_visualize = N.sum(dim=2, keepdim=False)

plt.imshow(N_visualize, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j,i, N_visualize[i,j].item(), ha='center', va='top', color='gray')

plt.show()
