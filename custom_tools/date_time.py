from langchain_core.tools import tool


@tool
def get_system_date_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current system date and time in the specified format."""
    from datetime import datetime

    currenttime = datetime.now()
    formattedtime = currenttime.strftime(format)

    return formattedtime
