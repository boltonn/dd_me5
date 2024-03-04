from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError
    
    @abstractmethod
    def unload(self):
        raise NotImplementedError
    
    @abstractmethod
    def infer(self):
        raise NotImplementedError