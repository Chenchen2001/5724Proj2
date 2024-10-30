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
    """
      Compute the dot product result of two vectors.

      Args:
        vec1, vec2: Two vectors in list style to be dotted.

      Return:
        The dot product result of two vectors.
    """
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

  def scalar_multiply(self, scalar: float, vec: list) -> list:
    """
      Multiply each element of a vector by a scalar.

      Args:
        scalar: The number used to multiply with the vector.
        vec: The vector to be scalared.

      Returns:
        The multiplied vector.
    """
    return [scalar * v for v in vec]

  def vector_add(self, vec1: list, vec2: list) -> list:
    """
      Add two vectors element-wise.

      Args:
        vec1, vec2: Two vectors to be added together element by element.
      
      Returns:
        Vector result of addition execution.
    """
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]
  
  def normalize(self, x: list) -> list:
    """
      Normalizes a vector to have unit length.

      Args:
        x (list): The input vector to normalize.
        
      Returns:
        list: The normalized vector with unit length.
      """
    norm = sum(i**2 for i in x) ** 0.5
    return [i / norm for i in x] if norm != 0 else x

  def train(self, data: list) -> None:
    """
      Train the margin perceptron model.

      Args:
        data: Dataset, including features in the front dims element and label as the final element.
    """
    for epoch in range(self.epochs):
      updates = 0
      for point in data:
        x = point[:-1]  # Features
        y = point[-1]   # Label
        x = self.normalize(x)
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

  def calculate_margin(self, data: list[list]) -> float:
    """
      Calculate the margin of the model based on the data.

      Args:
        data: Dataset, including features in the front dims element and label as the final element.
      
      Return:
        The margin(minimum) of the model on the dataset.
    """
    margins = [line[-1] * (self.dot_product(self.w, self.normalize(line[:-1])) + self.b) for line in data]
    return min(margins)