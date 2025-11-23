from typing import List

from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    print(f"xkn Validating user {user_id} with addresses: {addresses}")
    return True


llm = ChatOllama(
    model="gpt-oss:20b-cloud",
    validate_model_on_init=True,
    temperature=0,
).bind_tools([validate_user])

result = llm.invoke(
    "Could you validate user 123? They previously lived at "
    "123 Fake St in Boston MA and 234 Pretend Boulevard in "
    "Houston TX."
)

if isinstance(result, AIMessage) and result.tool_calls:
    print("Tool calls made during the LLM invocation:")
    print("tool call result is",result.tool_calls)

    for call in result.tool_calls:
        print(call)
        # invoke the tool (support both dict-style and object-style tool calls)
        tool_result = None
        if isinstance(call, dict):
            tool_obj = call.get("tool") or call.get("tool_name") or call.get("name")
            tool_input = call.get("tool_input") or call.get("input") or call.get("args") or {}

            # If tool is a string name, try to resolve it from the LLM's bound tools
            if isinstance(tool_obj, str):
                tool_callable = None
                if hasattr(llm, "tools"):
                    tool_callable = llm.tools.get(tool_obj)
                # fallback: try to resolve a global function with that name
                if tool_callable is None:
                    tool_callable = globals().get(tool_obj)
                if tool_callable is None:
                    print(f"Could not resolve tool callable for name: {tool_obj}")
                else:
                    if callable(tool_callable):
                        tool_result = tool_callable(**tool_input)
                    elif hasattr(tool_callable, "invoke"):
                        # Some tool wrappers expect a single 'input' argument rather than **kwargs.
                        try:
                            tool_result = tool_callable.invoke(**tool_input)
                        except TypeError:
                            try:
                                tool_result = tool_callable.invoke(tool_input)
                            except Exception as e:
                                print("Tool invoke failed:", e)
                    else:
                        print(f"Resolved tool is not callable or invokable: {tool_callable}")

            # If tool is already a callable/function
            elif callable(tool_obj):
                tool_result = tool_obj(**(tool_input or {}))

            # If tool is an object with an invoke method
            elif hasattr(tool_obj, "invoke"):
                tool_result = tool_obj.invoke(**(tool_input or {}))

            else:
                print("Unrecognized tool entry in tool call:", tool_obj)

        else:
            # original object-style calls (has attributes)
            tool_result = call.tool.invoke(**call.tool_input)

        print("tool result is", tool_result)
else:
    print("No tool calls were made during the LLM invocation.")
    
