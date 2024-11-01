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
        """
        Initialize the MarginPerceptron instance with given parameters.

        Args:
            dimension: The dimension of each data point.
            radius: The radius used to calculate the number of epochs.
            input: The input dataset as a list of data points.
            label: The labels for each data point.
        """
        self.input = input
        self.label = label
        self.dimension = dimension
        self.w = [0.0] * int(dimension)  # Initialize weight vector with zeros
        self.gamma_guess = radius # initial gamma
        self.epochs = self.max_iteration(radius, radius)

    def get_weights(self) -> list | float:
        """
        Get the current weight vector.

        Returns:
            The weight vector as a list of floats.
        """
        return self.w

    def max_iteration(self, rad: float, gamma_guess: float) -> int:
        """
        Calculate the maximum number of epochs for training.

        Args:
            rad: Radius of the dataset.
            gamma_guess: Initial guess for the margin.

        Returns:
            The maximum number of training epochs.
        """
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
        assert len(vec1) == len(vec2)
        list_len = len(vec1)
        for i in range(list_len):
            value += float(vec1[i]) * float(vec2[i])
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

    def norm(self, x: list) -> float:
        """
        Compute the Euclidean norm of a vector.

        Args:
            x: The vector to calculate the norm for.

        Returns:
            The Euclidean norm of the vector.
        """
        return math.sqrt(sum(i**2 for i in x))

    def iterate(self):
        """
        Iterate over the dataset to find a violation point (a point that is misclassified or violates the margin).

        Returns:
            The index of the first violation point found, or -2 if no violation point is found.
        """
        violation_point_index = -2
        for i, point in enumerate(self.input):
            point_label = self.label[i]
            dot_product = self.dot_product(self.w, point)
            # Check if weight vector is still at its initial value (all zeros)
            if self.w == [0.0] * int(self.dimension):
                violation_point_index = i
                break
            else:
                # Determine the predicted label based on dot product
                if dot_product < 0:
                    predict_label = -1
                else:
                    predict_label = 1
                # Calculate the distance to the margin
                norm = self.norm(self.w)
                distance = abs(dot_product) / norm
                # Check if the point violates the margin or is misclassified
                if distance < (self.gamma_guess / 2.0) or (predict_label * point_label < 0):
                    violation_point_index = i
                    break
        return violation_point_index

    def train(self):
        """
        Train the perceptron using the margin perceptron algorithm.

        Returns:
            True if training continues, False if no more violation points are found.
        """
        for _ in range(self.epochs):
            violation_point_index = self.iterate()
            if violation_point_index > -1:
                # Update weight vector with the violation point
                print("Find violation point index: ", violation_point_index)
                print("Current w: ", self.w)
                for j in range(self.dimension):
                    self.w[j] += self.label[violation_point_index] * float(self.input[violation_point_index][j])
                print("New w: ", self.w)
                print("------------------------------------")
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
            margin = (dot_product * self.label[i]) / norm_w
            margins.append(margin)

        if margins:
            return min(margins)
        else:
            return 0.0
