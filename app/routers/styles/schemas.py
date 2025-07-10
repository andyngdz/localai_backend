from pydantic import BaseModel

from app.predefined_styles.schemas import StyleItem


class StylesResponse(BaseModel):
    """
    Represents the response from the styles endpoint.
    """

    fooocus: list[StyleItem]
    sai: list[StyleItem]
