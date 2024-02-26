from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def infer(self):
        raise NotImplementedError