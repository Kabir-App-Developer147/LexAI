from core.llm import llm

def main():
    print("LexAI - Your Local Business Assistant")
    print("Type 'quit' to exit\n")
    
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