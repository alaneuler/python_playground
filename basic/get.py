class A:
    def __init__(self):
        self.self = "asdf"


a = A()
print(getattr(a, "self"))
