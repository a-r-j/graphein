from pydantic import BaseModel


class HiCConfig(BaseModel):
    format: str = "cooler"
