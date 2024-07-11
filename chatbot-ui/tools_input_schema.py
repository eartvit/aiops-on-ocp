from pydantic.v1 import BaseModel, Field

class ToolsInputSchema(BaseModel):
    """Input schema model for the tools"""
    input_parameters: str = Field(description="The input parameters as a dictionary that uses double quotes for the parameter names. Do NOT enclose the dictionary in any additional quotes.")
