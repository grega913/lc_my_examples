from tutorials.pdf_qa import  createAChain


def execute_function_based_on_input(rag_chain):
  """
  Accepts user input from the terminal and executes the corresponding function.

  Args:
    function_dict: A dictionary mapping input strings to functions.
  """

  while True:
    user_input = input("Enter a command (or 'quit' to exit): ")
    if user_input == 'quit':
      break



    results=rag_chain.invoke({"input": user_input })
    print(results.answer)


if __name__=="__main__":

   #chain = createAChain(fName="data/kd-t1-diabetes-tn.pdf")
   chain = createAChain(fName="data/nke-10k-2023.pdf")
   execute_function_based_on_input(chain)