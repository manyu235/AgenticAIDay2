from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the LLM
llm = ChatOllama(model="gemma3:latest")

# Define the prompt template for client email drafting
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a project manager drafting a client email. Your task is to write professional emails summarizing project progress.

Requirements:
1. Use a formal and professional tone
2. Include placeholders for [Client Name], [Project Name], and [Deadline]
3. Summarize project progress
4. Mention upcoming milestones
5. Request feedback from the client
6. Add a bullet list of action items
7. Include a subject line
8. Avoid sensitive data
9. No markdown formatting in the response

Output Format:
- Subject line
- Email body with professional greeting
- Project progress summary
- Upcoming milestones
- Feedback request
- Action items as bullet points
- Professional closing"""),
    ("human", "Draft an email based on the following details:\n\n{email_details}")
])

def draft_client_email(email_details):
    """
    Draft a professional client email based on provided details.
    
    Args:
        email_details (str): Details about the project, progress, and milestones
        
    Returns:
        str: The drafted email
    """
    try:
        # Format the prompt with the email details
        formatted_prompt = prompt_template.format(email_details=email_details)
        
        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        
        return response.content
    except Exception as e:
        return f"Error drafting email: {str(e)}"

def main():
    """Main function to run the client email drafter interactively."""
    print("Client Email Drafter")
    print("Draft professional emails to clients summarizing project progress.")
    
    while True:
        user_input = input("\nEnter 'draft' to create an email, or 'quit' to exit: ").strip().lower()
        
        if user_input == 'quit':
            print("Exiting.")
            break
        
        if user_input == 'draft':
            print("Provide details for the email.")
            print("Enter project information, progress updates, and milestones.")
            print("When finished, enter '###END###' on a new line:")
            
            lines = []
            while True:
                line = input()
                if line.strip() == '###END###':
                    break
                lines.append(line)
            
            email_details = '\n'.join(lines)
            
            if not email_details.strip():
                print("Error: No details provided. Please try again.")
                continue
            
            print("\nDrafting email...")
            
            # Draft the email
            email = draft_client_email(email_details)
            
            print("\nDrafted Email:")
            print(email)
        else:
            print("Invalid input. Please enter 'draft' or 'quit'.")

if __name__ == "__main__":
    main()
