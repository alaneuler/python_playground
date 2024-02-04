from abc import ABC, abstractmethod


class MyABC(ABC):
    @abstractmethod
    def area(self):
        pass


ma = MyABC()
