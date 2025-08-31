import json
import datetime
from app import llama3_chat_model, route_llm_output

def run_case(user_text: str):
    print("="*80)
    print("User query:", user_text)
    llm_out = llama3_chat_model(user_text)
    print("Raw LLM output:", llm_out)
    logs = route_llm_output(llm_out)
    print("Function call made?:", logs["is_function_call"])
    print("Called function:", logs["called_function"])
    print("Function args:", logs["function_args"])
    print("Tool output:", logs["tool_output"])
    print("Final assistant response:", logs["final_response"])
    print("Timestamp:", datetime.datetime.now().isoformat())
    print("="*80 + "\n")

def main():
    # 1) Math query
    run_case("Hey, can you calculate 2+2?")
    # 2) arXiv search
    run_case("Find arXiv papers on quantum entanglement.")
    # 3) Normal query
    run_case("Tell me a fun fact about black holes.")

if __name__ == "__main__":
    main()
