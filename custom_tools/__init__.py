from custom_tools.image_tool import generate_image_tool, generate_image_and_save_tool
from custom_tools.date_time import get_system_date_time
from custom_tools.web_tools import search_arxiv, search_web, search_wiki

queai_tools = [
    search_web,
    search_wiki,
    search_arxiv,
    get_system_date_time,
]

image_tools = [generate_image_tool, generate_image_and_save_tool]
