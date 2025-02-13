# from decouple import config
# from pydantic_settings import BaseSettings


# class Settings(BaseSettings):
#     GENAI_API_KEY: str = "AIzaSyBwzbuL30EsbczQb9rWyBFVGB9S2rKG5y4"
# settings = Settings()

from decouple import config
class Settings:
    GENAI_API_KEY: str = config("GENAI_API_KEY")
settings = Settings()
