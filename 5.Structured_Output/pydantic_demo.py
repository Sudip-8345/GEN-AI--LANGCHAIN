from pydantic import BaseModel, EmailStr, Field
from typing import Optional
import json

class Person(BaseModel):
    name: str = 'sudip'
    gender: str = 'male'
    email: EmailStr
    age: Optional[int] = None
    height : float = Field(gt=155, lt=175, default = 160, description='the height of the person')

# Corrected: use a dictionary, not a set
person1_data = {'name': 'Aditi', 'gender': 'female', 'email': 'Aditi@gmail.com','height':160}
person1 = Person(**person1_data)

person1_dict = dict(person1)
person1_json = person1.model_dump_json()
print(json.loads(person1_json)['name'])