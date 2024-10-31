import math
class MarginPerceptron:
    """
    This class is designed for implementing the margin perceptron algorithm
    using the dataset given by the project of CMSC5724 Project 2 Margin Perceptron.

    Attributes:
	dimension: The dimension for each data point in the dataset.
    margin_threshold: The minimum margin required for correctly classified points.
    epochs: The maximum number of training iterations over the dataset. The training may stop earlier than epoch.
    """
    def __init__(self, dimension: int, radius: float, input: list, label: list):
        self.input = input
        self.label = label
        self.dimension = dimension
        self.w = [0.0] * int(dimension)
        self.gamma_guess = radius
        self.epochs = self.max_iteration(radius, radius)


    def get_weights(self) -> list | float:
        return self.w

    def max_iteration(self, rad, gamma_guess):
        self.epochs = math.ceil(12 * (rad ** 2) / (gamma_guess ** 2))
        return self.epochs

    def dot_product(self, vec1: list, vec2: list) -> float:
        """
      Compute the dot product result of two vectors.

      Args:
        vec1, vec2: Two vectors in list style to be dotted.

      Return:
        The dot product result of two vectors.
    """
        value = 0
        list_len = len(vec1)
        for i in range(list_len):
            value += float(vec1[i])* float(vec2[i])
        return value

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

    def norm(self, x):
        return math.sqrt(sum(i**2 for i in x))

    def iterate(self):
        violation_point_index = -2
        for i, point in enumerate(self.input):
            point_label = self.label[i]
            dot_product = self.dot_product(self.w, point)
            if self.w == [0.0] * int(self.dimension):
                violation_point_index = i
                break
            else:
                if dot_product < 0:
                    predict_label = -1
                else:
                    predict_label = 1
                norm = self.norm(self.w)
                distance = abs(dot_product) / norm
                if distance < (self.gamma_guess / 2.0) or (predict_label * point_label < 0):
                    violation_point_index = i
                    break
        return violation_point_index

    def train(self):
        for i in range(self.epochs):
            violation_point_index = self.iterate()
            if violation_point_index > -1:
                print("Find violation point index: ", violation_point_index)
                print("Current w: ", self.w)
                for j in range(self.dimension):
                    self.w[j] += self.label[violation_point_index] * float(self.input[violation_point_index][j])
                print("New w: ", self.w)
                print('\n\n')
            else:
                return False
            return True

    def calculate_margin(self):
        """
        Calculate the margin for the current weight vector.

        Returns:
            The minimum margin over all data points.
        """
        margins = []
        for i, point in enumerate(self.input):
            dot_product = self.dot_product(self.w, point)
            norm_w = self.norm(self.w)
            if norm_w == 0:
                continue
            margin = (dot_product *self.label[i])/ norm_w
            margins.append(margin)

        if margins:
            return min(margins)
        else:
            return 0.0
