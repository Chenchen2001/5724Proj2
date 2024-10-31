import math

class MarginPerceptron:
  """
    This class is designed for implementing the margin perceptron algorithm
    using the dataset given by the project of CMSC5724 Project 2 Margin Perceptron.

    Attributes:
			dataIntro: The introduction of the dataset will be given to.
      dataset: The dataset for training.
  """
  def __init__(self, dataIntro: list, dataset: list[list]):
    self.dim, self.pts, self.radius = tuple(dataIntro)
    self.dataset = dataset
    self.w = [0.0] * int(self.dim)

  def get_weights(self) -> list:
    return self.w

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
  
  def vector_subtract(self, vec1: list, vec2: list) -> list:
    """
      Substract two vectors element-wise.

      Args:
        vec1, vec2: Two vectors to be substracted element by element, vec1 - vec2.
      
      Returns:
        Vector result of substraction execution.
    """
    return [v1 - v2 for v1, v2 in zip(vec1, vec2)]

  def train(self) -> None:
    """
      Train the margin perceptron model.
    """
    R = max([math.sqrt(sum([element**2 for element in line[:-1]])) for line in self.dataset])
    gamma_guess = R
    while True:
      updates = 0
      max_iterations = math.ceil((12 * R**2) / (gamma_guess**2))  # 12R^2 / gamma_guess^2
      iteration = 0
      for iter in range(max_iterations):
        iteration += 1
        updates = 0
        for point in self.dataset:
          x = point[:-1]  # Features
          y = point[-1]   # Label
          margin = y * self.dot_product(self.w, x)
          # If margin is less than gamma_guess, update weights
          if margin < gamma_guess:
            if y == 1:
              print("update, add")
              self.w = self.vector_add(self.w, x)
            else:
              print("update, subtract")
              self.w = self.vector_subtract(self.w, x)
            updates += 1
        print(f"In the iter {iter}/{max_iterations}, executed {updates} updates.")
        # Self-termination: if no updates, we have converged
        if updates == 0:
            print(f"Converged with gamma_guess = {gamma_guess}.")
            return
      # Forced-termination: if max iterations reached without convergence, halve gamma_guess
      print(f"Forced termination with gamma_guess = {gamma_guess}. Reducing gamma_guess and retrying.")
      gamma_guess /= 2  # Halve gamma_guess as per the algorithm

  def calculate_margin(self) -> float:
    """
      Calculate the margin of the model based on the data.

      Args:
        data: Dataset, including features in the front dims element and label as the final element.
      
      Return:
        The margin(minimum) of the model on the dataset.
    """
    margins = [line[-1] * (self.dot_product(self.w, line[:-1])) for line in self.dataset]
    return min(margins)