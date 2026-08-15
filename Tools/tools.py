from langchain_community.tools import DuckDuckGoSearchRun, ShellTool


# DuckDuckgo


search_tool = DuckDuckGoSearchRun()

res = search_tool.invoke('Bitcoin news')

print(res)


#Shell
shell_tool = ShellTool()

res = shell_tool.invoke('whoami')

print(res)
