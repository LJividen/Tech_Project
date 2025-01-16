import pandas as pd
import numpy as np

# Generate random numerical matrix (10k x 9)
homogeneous_data = pd.DataFrame({
    f"col_{i}": np.random.randint(0, 100, size=10_000) for i in range(9)
})

print(homogeneous_data.shape)  # Should print (10000, 9)
df = pd.DataFrame(homogeneous_data)
df.to_csv("homogeneous_data.csv", index=False)
