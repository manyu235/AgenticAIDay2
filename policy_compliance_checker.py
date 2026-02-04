from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json

# Initialize the LLM
llm = ChatOllama(model="gemma3:latest")

# Define the prompt template for policy compliance checking
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an HR compliance auditor. Your task is to review HR policy drafts and identify compliance issues.

Requirements:
1. Identify missing compliance clauses
2. Flag ambiguous language
3. Suggest improvements
4. Cite policy references where applicable
5. Do not invent compliance rules; base suggestions only on provided context
6. Output must be in JSON format with keys: issues, severity, recommendations
7. Severity levels: Critical, High, Medium, Low

Output Format (JSON):
{{
  "issues": ["issue1", "issue2"],
  "severity": ["Critical", "Medium"],
  "recommendations": ["recommendation1", "recommendation2"]
}}

Important: Base all suggestions on the provided context. Do not make up compliance rules."""),
    ("human", "Review the following HR policy text:\n\n{policy_text}")
])

def check_policy_compliance(policy_text):
    """
    Review a policy draft and identify compliance issues.
    
    Args:
        policy_text (str): The HR policy text to review
        
    Returns:
        dict: JSON object with issues, severity, and recommendations
    """
    try:
        # Invoke the LLM with the prompt template
        response = llm.invoke(prompt_template.format_messages(policy_text=policy_text))
        
        # Try to parse the response as JSON
        try:
            # Extract JSON from response
            content = response.content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                raise json.JSONDecodeError("No JSON found", content, 0)
        except json.JSONDecodeError:
            # If response is not valid JSON, return it as-is with a note
            return {
                "issues": ["Response parsing error"],
                "severity": ["N/A"],
                "recommendations": [response.content]
            }
    except Exception as e:
        return {
            "issues": [f"Error checking compliance: {str(e)}"],
            "severity": ["Critical"],
            "recommendations": ["Please try again or check your input."]
        }

def main():
    """Main function to run the policy compliance checker interactively."""
    print("Policy Compliance Checker")
    print("Review HR policy drafts for compliance issues and ambiguous language.")
    
    while True:
        user_input = input("\nEnter 'review' to check a policy, or 'quit' to exit: ").strip().lower()
        
        if user_input == 'quit':
            print("Exiting.")
            break
        
        if user_input == 'review':
            print("Paste your HR policy text below.")
            print("When finished, enter '###END###' on a new line:")
            
            lines = []
            while True:
                line = input()
                if line.strip() == '###END###':
                    break
                lines.append(line)
            
            policy_text = '\n'.join(lines)
            
            if not policy_text.strip():
                print("Error: No policy text provided. Please try again.")
                continue
            
            print("\nReviewing policy for compliance...")
            
            # Check compliance
            result = check_policy_compliance(policy_text)
            
            print("\nCompliance Review Results:")
            print(json.dumps(result, indent=2))
        else:
            print("Invalid input. Please enter 'review' or 'quit'.")

if __name__ == "__main__":
    main()
