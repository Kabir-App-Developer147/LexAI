from core.llm import get_llm_manager

def main():
    print("LexAI - Your Local Business Assistant")
    print("Type 'quit' to exit\n")
    
    llm = get_llm_manager()
    history = []
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue
            
        response, history = llm.chat(user_input, history)
        print(f"\nLexAI: {response}\n")

if __name__ == "__main__":
    main()