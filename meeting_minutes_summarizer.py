from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the LLM
llm = ChatOllama(model="gemma3:latest")

# Define the prompt template for meeting minutes summarization
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a meeting assistant. Your task is to summarize meeting transcripts into structured decisions and action items.

Requirements:
1. Create sections: Decisions and Action Items
2. For each action item, include:
   - Owner (person responsible)
   - Deadline (when it's due)
   - Confidence score (how clear/certain the assignment is)
3. Output in Markdown format
4. Avoid adding items not in the transcript
5. Ensure clarity and accuracy
6. Do not invent information

Output Format (Markdown):
## Decisions
- Decision 1
- Decision 2

## Action Items
- Action item description (Owner: Name, Deadline: Date, Confidence: High/Medium/Low)
- Action item description (Owner: Name, Deadline: Date, Confidence: High/Medium/Low)

Important: Only extract information that is explicitly stated or clearly implied in the transcript."""),
    ("human", "Summarize the following meeting transcript:\n\n{transcript_text}")
])

def summarize_meeting_minutes(transcript_text):
    """
    Summarize a meeting transcript into decisions and action items.
    
    Args:
        transcript_text (str): The meeting transcript to summarize
        
    Returns:
        str: Structured Markdown summary with decisions and action items
    """
    try:
        # Format the prompt with the transcript text
        formatted_prompt = prompt_template.format(transcript_text=transcript_text)
        
        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        
        return response.content
    except Exception as e:
        return f"Error summarizing meeting: {str(e)}"

def main():
    """Main function to run the meeting minutes summarizer interactively."""
    print("Meeting Minutes Summarizer")
    print("Summarize meeting transcripts into decisions and action items.")
    
    while True:
        user_input = input("\nEnter 'summarize' to process a transcript, or 'quit' to exit: ").strip().lower()
        
        if user_input == 'quit':
            print("Exiting.")
            break
        
        if user_input == 'summarize':
            print("Paste your meeting transcript below.")
            print("When finished, enter '###END###' on a new line:")
            
            lines = []
            while True:
                line = input()
                if line.strip() == '###END###':
                    break
                lines.append(line)
            
            transcript_text = '\n'.join(lines)
            
            if not transcript_text.strip():
                print("Error: No transcript provided. Please try again.")
                continue
            
            print("\nSummarizing meeting transcript...")
            
            # Summarize the meeting
            summary = summarize_meeting_minutes(transcript_text)
            
            print("\nMeeting Summary:")
            print(summary)
        else:
            print("Invalid input. Please enter 'summarize' or 'quit'.")

if __name__ == "__main__":
    main()
