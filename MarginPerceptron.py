import math

class MarginPerceptron:
    """
    This class is designed for implementing the margin perceptron algorithm
    using the dataset given by the project of CMSC5724 Project 2 Margin Perceptron.
    Attributes:
        dimension: The dimension of each data point.
        radius: The radius used to calculate the number of epochs.
        input: The input dataset as a list of data points.
        label: The labels for each data point.
    """
    def __init__(self, dimension: int, radius: float, input: list, label: list):
        self.input = input
        self.label = label
        self.dimension = dimension
        self.w = [0.0] * int(dimension)  # Initialize weight vector with zeros
        self.gamma_guess = radius  # Initial gamma guess
        self.radius = radius  # Store initial radius
        self.epochs = self.max_iteration(radius, radius)

    def get_weights(self) -> list:
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

        return math.ceil(12 * (rad ** 2) / (gamma_guess ** 2))

    def dot_product(self, vec1: list, vec2: list) -> float:
        """
        Compute the dot product result of two vectors.
        Args:
            vec1, vec2: Two vectors in list style to be dotted.
        Return:
            The dot product result of two vectors.
        """
        return sum(float(v1) * float(v2) for v1, v2 in zip(vec1, vec2))

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
        for i, point in enumerate(self.input):
            point_label = self.label[i]
            dot_product = self.dot_product(self.w, point)
            # Determine the predicted label
            predict_label = 1 if dot_product >= 0 else -1
            # Calculate distance to margin
            norm_w = self.norm(self.w)
            distance = abs(dot_product) / norm_w if norm_w != 0 else 0

            # Check for violation: margin or misclassification
            if (distance < (self.gamma_guess / 2.0)) or (predict_label * point_label < 0):
                return i  # Violation point index
        return -1  # No violation point found

    def update_weights(self, index: int):
        """
        Update the weight vector based on the violation point.
        
        Returns:
            None
        """
        
        for j in range(self.dimension):
            self.w[j] += self.label[index] * float(self.input[index][j])

    def train(self):
        """
        Train the perceptron using the margin perceptron algorithm.

        Returns:
            True if training continues, False if no more violation points are found.
        """
        while True:
            for _ in range(self.epochs):
                violation_point_index = self.iterate()
                # Self-termination: If no violation point is found, training is complete
                if violation_point_index == -1:
                    print("Training completed with self-termination.")
                    return False
                # Update weights based on the violation point
                print(f"Violation point index: {violation_point_index}")
                print(f"Current weights: {self.w}")
                self.update_weights(violation_point_index)
                print(f"Updated weights: {self.w}")
                print("------------------------------------")

            # Forced-termination: Reduce gamma_guess and recompute epochs
            self.gamma_guess /= 2
            if self.gamma_guess <= 1e-8:
                print("Gamma guess is too small to continue training.")
                return False
            print(f"Forced termination: Reducing gamma_guess to {self.gamma_guess} and restarting training.")
            self.epochs = self.max_iteration(self.radius, self.gamma_guess)

    def calculate_margin(self):
        """
        Calculate the margin for the current weight vector.
        Returns:
            The minimum margin over all data points.
        """
        margins = []
        norm_w = self.norm(self.w)
        if norm_w == 0:
            return 0.0  # Avoid division by zero
        for i, point in enumerate(self.input):
            margin = (self.dot_product(self.w, point) * self.label[i]) / norm_w
            margins.append(margin)
        return min(margins) if margins else 0.0
