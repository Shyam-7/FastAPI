from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase

def get_sql_tool():
    """
    Returns a SQL tool.
    """
    db = SQLDatabase.from_uri("sqlite:///./test.db")
    return QuerySQLDataBaseTool(db=db)
