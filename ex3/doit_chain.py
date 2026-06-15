from pydantic import BaseModel,Field
from typing import Literal,Optional


class BashCmd(BaseModel):
    """Command to be executed in bash shell"""
    intent:Literal["execute_command","conversation","error"]
    command : Optional[str] =Field(description = "Command to be executed in bash shell , only populated if intent is execute_command")
    conversation : Optional[str] =Field(description = "Response to be returned if the query is a conversation , only populated if intent is conversation")
    error : Optional[str] =Field(description = "Response to be returned if the query is an error like impossible command , only populated if intent is error")
    
