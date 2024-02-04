class Parent(object):
    def __init__(self, value):
        print(hex(id(self.__dict__)))
        self.pname = "parent"
        self.value = value - 1


class Child(Parent):
    def __init__(self, value):
        print(hex(id(self.__dict__)))
        super().__init__(value)
        self.cname = "child"
        self.value = value


c = Child(5)
print(c)
p = super(c.__class__, c)
print(p)


# Information of `id` and `name` is stored in `__annotations__` of special variables
class My1:
    id: int
    name: str


class My2:
    def __init__(self, id, name):
        self.id = id
        self.name = name


m1 = My1()
print(m1)
m2 = My2(0, "alaneuler")
print(m2)
