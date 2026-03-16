import numpy as np

class DataAugmentationFactory:
    @staticmethod
    def get_augmentation_method(method_name):
        if method_name == "random_copy":
            return RandomCopyAugmentation
        elif method_name == "random_noise":
            return RandomNoiseAugmentation
        else:
            raise ValueError(f"Unknown augmentation method: {method_name}")

class RandomCopyAugmentation:
    def __init__(self, num_copies=2):
        self.num_copies = num_copies

    def augment(self, data, labels):
        augmented_data = []
        augmented_labels = []
        for d, l in zip(data, labels):
            for _ in range(self.num_copies):
                augmented_data.append(d.copy())
                augmented_labels.append(l.copy())
        return np.array(augmented_data), np.array(augmented_labels)

class RandomNoiseAugmentation:
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def augment(self, data, labels):
        augmented_data = []
        augmented_labels = []
        for d, l in zip(data, labels):
            noise = np.random.normal(0, self.noise_level, d.shape)
            augmented_data.append(d + noise)
            augmented_labels.append(l)
        return np.array(augmented_data), np.array(augmented_labels)
