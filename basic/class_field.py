class Parent:
    def __init__(self, value):
        self.value = value


class Child(Parent):
    def __init__(self, value):
        self.value = value


c = Child(5)
print(c)
