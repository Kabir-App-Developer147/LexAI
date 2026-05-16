import ollama

def chat(user_message, conversation_history):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": "You are LexAI, a helpful assistant for small businesses and professionals in India. Be concise and practical. Always consider Indian laws, GST, and local context where relevant."},
            *conversation_history
        ]
    )
    
    reply = response['message']['content']
    conversation_history.append({
        "role": "assistant", 
        "content": reply
    })
    
    return reply, conversation_history

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
            
        response, history = chat(user_input, history)
        print(f"\nLexAI: {response}\n")

if __name__ == "__main__":
    main()