from pydantic import BaseModel

class APIOutput(BaseModel):
    emotion: str
    time_elapsed: float
    time_elapsed_preprocess: float