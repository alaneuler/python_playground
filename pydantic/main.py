from typing import Optional

from pydantic import BaseModel


class MyModel(BaseModel):
    id: int
    name: Optional[str] = None


mm = MyModel(id=1, name="alaneuler")
print(mm)
mm = MyModel(id=1)
print(mm)
