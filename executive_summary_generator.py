from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the LLM
llm = ChatOllama(model="gemma3:latest")

# Define the prompt template with role-based context and specific requirements
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analyst. Your task is to summarize quarterly performance reports into executive summaries.

Requirements:
1. Create a 150-word executive summary
2. Include a table of key metrics (Revenue, EPS, Growth %)
3. Add a section listing top risks and opportunities
4. Ensure all data is factual and derived only from the provided context
5. Do not use markdown formatting in the response
6. Cite sources if needed

Output Format:
- Executive Summary (150 words)
- Key Metrics Table
- Risks and Opportunities Section"""),
    ("human", "Summarize the following quarterly report:\n\n{report_text}")
])

def generate_executive_summary(report_text):
    """
    Generate an executive summary from a quarterly performance report.
    
    Args:
        report_text (str): The full quarterly report text
        
    Returns:
        str: The generated executive summary
    """
    try:
        # Format the prompt with the report text
        formatted_prompt = prompt_template.format(report_text=report_text)
        
        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        
        return response.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def main():
    """Main function to run the executive summary generator interactively."""
    print("Executive Summary Generator")
    print("Enter your quarterly report text or 'quit' to exit.")
    
    while True:
        user_input = input("\nEnter 'paste' to input a multi-line report, or 'quit' to exit: ").strip().lower()
        
        if user_input == 'quit':
            print("Exiting.")
            break
        
        if user_input == 'paste':
            print("Paste your quarterly report text below.")
            print("When finished, enter '###END###' on a new line:")
            
            lines = []
            while True:
                line = input()
                if line.strip() == '###END###':
                    break
                lines.append(line)
            
            report_text = '\n'.join(lines)
            
            if not report_text.strip():
                print("Error: No report text provided. Please try again.")
                continue
            
            print("\nGenerating executive summary...")
            
            # Generate the summary
            summary = generate_executive_summary(report_text)
            
            print("\nResult:")
            print(summary)
        else:
            print("Invalid input. Please enter 'paste' or 'quit'.")

if __name__ == "__main__":
    main()
