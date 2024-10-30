class MarginPerceptron:
  """
    This class is designed for implementing the margin perceptron algorithm
    using the dataset given by the project of CMSC5724 Project 2 Margin Perceptron.

    Attributes:
			dimension: The dimension for each data point in the dataset.
      margin_threshold: The minimum margin required for correctly classified points. 
      epochs: The maximum number of training iterations over the dataset. The training may stop earlier than epoch.
  """
  def __init__(self, dimension: int = 2, margin_threshold: float = 1.0, epochs: int =10):
    self.w = [0.0] * int(dimension)  # Initialize weights as a list of zeros
    self.b = 0.0  # Initialize bias to zero
    self.margin_threshold = margin_threshold
    self.epochs = epochs

  def get_params(self, type: str) -> list | float:
    if(type == "weight" or type == "w"):
      return self.w
    if(type == "bias" or type == "b"):
      return self.b

  def dot_product(self, vec1: list, vec2: list) -> float:
    """Compute the dot product result of two vectors."""
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

  def scalar_multiply(self, scalar: float, vec: list) -> list:
    """Multiply each element of a vector by a scalar."""
    return [scalar * v for v in vec]

  def vector_add(self, vec1: list, vec2: list) -> list:
    """Add two vectors element-wise."""
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]

  def train(self, data: list) -> None:
    for epoch in range(self.epochs):
      updates = 0
      for point in data:
        x = point[:-1]  # Features
        y = point[-1]   # Label
        margin = y * (self.dot_product(self.w, x) + self.b)
                
        # If margin is less than the threshold, update weights and bias
        if margin < self.margin_threshold:
          self.w = self.vector_add(self.w, self.scalar_multiply(y, x))
          self.b += y
          updates += 1
            
      # Stop if no updates are made in the current epoch
      if updates == 0:
        print(f"Converged after {epoch+1} epochs.")
        break

  def calculate_margin(self, data: list) -> float:
    margins = [point[-1] * (self.dot_product(self.w, point[:-1]) + self.b) for point in data]
    return min(margins)