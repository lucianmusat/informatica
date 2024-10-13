from langchain.callbacks.base import BaseCallbackHandler
from html_templates import bot_template


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(bot_template.replace("{{MSG}}", self.text), unsafe_allow_html=True)
