from typing import List
from pydantic import BaseModel, Field
from enum import Enum


class Response(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]
    response: str


class UserResponse(BaseModel):
    response: str
    extracted_info: str


class EconsultType(str, Enum):
    # TODO look at optimizing to_list initialize
    # when adding new types also add them to the to_list method
    address_change = "address_change"
    medication = "medication"
    insurance = "insurance"

    @staticmethod
    def to_list() -> List[str]:
        return ["address_change", "medication", "insurance"]
