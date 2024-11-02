import math

class MarginPerceptron:
    def __init__(self, dimension: int, radius: float, input: list, label: list):
        self.input = input
        self.label = label
        self.dimension = dimension
        self.w = [0.0] * int(dimension)  # Initialize weight vector with zeros
        self.gamma_guess = radius  # Initial gamma guess
        self.radius = radius  # Store initial radius
        self.epochs = self.max_iteration(radius, radius)

    def get_weights(self) -> list:
        """Return the current weight vector."""
        return self.w

    def max_iteration(self, rad: float, gamma_guess: float) -> int:
        """Calculate maximum number of epochs based on the radius and gamma guess."""
        return math.ceil(12 * (rad ** 2) / (gamma_guess ** 2))

    def dot_product(self, vec1: list, vec2: list) -> float:
        """Compute dot product of two vectors."""
        return sum(float(v1) * float(v2) for v1, v2 in zip(vec1, vec2))

    def norm(self, x: list) -> float:
        """Compute Euclidean norm of a vector."""
        return math.sqrt(sum(i**2 for i in x))

    def iterate(self):
        """Find a violation point in the dataset."""
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
        """Update the weight vector based on the violation point."""
        for j in range(self.dimension):
            self.w[j] += self.label[index] * float(self.input[index][j])

    def train(self):
        """Train the perceptron with self-termination and forced-termination mechanisms."""
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
        """Calculate the minimum margin of the current weight vector."""
        margins = []
        norm_w = self.norm(self.w)
        if norm_w == 0:
            return 0.0  # Avoid division by zero
        for i, point in enumerate(self.input):
            margin = (self.dot_product(self.w, point) * self.label[i]) / norm_w
            margins.append(margin)
        return min(margins) if margins else 0.0
