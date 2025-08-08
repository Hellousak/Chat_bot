while True:
    user_input = input("You:")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye! Have a great day!")
        break
    response = chain.invoke({"input": user_input, "history": history})
    print(f"Albert: {response}")
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response))
