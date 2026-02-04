from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json

# Initialize the LLM
llm = ChatOllama(model="gemma3:latest")

# Define the prompt template for market analysis brief generation
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a market research analyst. Your task is to generate market analysis briefs from news articles and internal reports.

Requirements:
1. Include a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)
2. Highlight top 3 trends with citations
3. Provide a narrative summary
4. No fabricated data; ensure all trends are supported by sources
5. All information must be derived from the provided context
6. Output in JSON format with keys: SWOT, trends, citations, narrative_summary

Output Format (JSON):
{{
  "SWOT": {{
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "opportunities": ["opportunity1", "opportunity2"],
    "threats": ["threat1", "threat2"]
  }},
  "trends": [
    {{"trend": "Trend description", "citation": "Source reference"}},
    {{"trend": "Trend description", "citation": "Source reference"}},
    {{"trend": "Trend description", "citation": "Source reference"}}
  ],
  "citations": ["Source 1", "Source 2", "Source 3"],
  "narrative_summary": "Comprehensive summary text..."
}}

Important: Base all analysis on the provided documents. Do not invent data or trends."""),
    ("human", "Generate a market analysis brief from the following documents:\n\n{documents}")
])

def generate_market_analysis(documents):
    """
    Generate a market analysis brief from provided documents.
    
    Args:
        documents (str): News articles and internal reports
        
    Returns:
        dict: JSON object with SWOT, trends, citations, and narrative summary
    """
    try:
        # Invoke the LLM with the prompt template
        response = llm.invoke(prompt_template.format_messages(documents=documents))
        
        # Try to parse the response as JSON
        try:
            # Try to extract JSON from the response
            content = response.content
            # Find JSON object in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                raise json.JSONDecodeError("No JSON found", content, 0)
        except json.JSONDecodeError as e:
            # If response is not valid JSON, return the raw response
            return {
                "SWOT": {
                    "strengths": [],
                    "weaknesses": [],
                    "opportunities": [],
                    "threats": []
                },
                "trends": [],
                "citations": [],
                "narrative_summary": response.content,
                "error": f"JSON parse error: {str(e)}"
            }
    except Exception as e:
        return {
            "SWOT": {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": []
            },
            "trends": [],
            "citations": [],
            "narrative_summary": f"Error generating analysis: {str(e)}"
        }

def main():
    """Main function to run the market analysis brief generator interactively."""
    print("Market Analysis Brief Generator")
    print("Generate market analysis briefs from news articles and internal reports.")
    
    while True:
        user_input = input("\nEnter 'analyze' to generate a brief, or 'quit' to exit: ").strip().lower()
        
        if user_input == 'quit':
            print("Exiting.")
            break
        
        if user_input == 'analyze':
            print("Paste your documents (news articles, reports, etc.) below.")
            print("When finished, enter '###END###' on a new line:")
            
            lines = []
            while True:
                line = input()
                if line.strip() == '###END###':
                    break
                lines.append(line)
            
            documents = '\n'.join(lines)
            
            if not documents.strip():
                print("Error: No documents provided. Please try again.")
                continue
            
            print("\nGenerating market analysis brief...")
            
            # Generate the analysis
            result = generate_market_analysis(documents)
            
            print("\nMarket Analysis Brief:")
            print(json.dumps(result, indent=2))
        else:
            print("Invalid input. Please enter 'analyze' or 'quit'.")

if __name__ == "__main__":
    main()
