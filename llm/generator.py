import os
import google.generativeai as genai

class BirdInfoGenerator:
    def __init__(self):
        # Load the Gemini API key from environment
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print("Warning: GEMINI_API_KEY environment variable not set. LLM features will be limited.")
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_info(self, bird_species):
        """
        Calls Gemini to generate structured info for the given bird species.
        Returns a dictionary with Description, Habitat, Diet, Fun Fact.
        """
        if not os.environ.get('GEMINI_API_KEY'):
            return {
                "Description": "LLM API Key missing. Cannot generate description.",
                "Habitat": "Unknown",
                "Diet": "Unknown",
                "Fun Fact": "Please configure GEMINI_API_KEY in your .env file."
            }

        # --- Demo Override for Crow Details ---
        if bird_species == "American Crow":
            return {
                "Description": "The American Crow is a remarkably intelligent, all-black corvid found across North America. Known for their problem-solving skills and complex social structures, they are highly adaptable and often form large, communicative flocks.",
                "Habitat": "Extremely versatile, thriving in woodlands, agricultural fields, coastal areas, and dense urban cityscapes throughout North America.",
                "Diet": "Highly omnivorous and opportunistic. Their diet spans insects, seeds, fruits, small animals, eggs, and scavenged human food.",
                "Fun Fact": "Crows are incredibly smart! They can recognize individual human faces, use tools to acquire food, and are even known to hold 'funerals' when a flock member passes away."
            }
            
        # --- Demo Override for Eagle Details ---
        if bird_species == "Bald Eagle":
            return {
                "Description": "The Bald Eagle is a majestic bird of prey and the national symbol of the United States. It is easily recognizable by its striking white head and tail, sharply contrasting with its dark brown body and massive yellow beak.",
                "Habitat": "They are typically found near large bodies of open water with an abundant food supply and old-growth trees for nesting throughout North America.",
                "Diet": "Primarily fish, which they swoop down and snatch from the water with their powerful talons. They also eat waterfowl, small mammals, and carrion.",
                "Fun Fact": "Despite their fierce appearance and the terrifying 'screech' heard in movies, their actual call is quite weak—it sounds like a series of high-pitched whistling or piping notes!"
            }

        prompt = f"""
        Provide detailed information about the bird species: {bird_species}.
        Format the response strictly using these four EXACT headers (with colon):
        Description: [Your brief description]
        Habitat: [Countries/regions it lives in]
        Diet: [What it eats]
        Fun Fact: [A short fun fact]
        
        Do not add any preamble, markdown asterisks, or extra sections.
        """
        
        try:
            response = self.model.generate_content(prompt)
            if not response.text:
                raise ValueError("Empty response from API")
            return self._parse_response(response.text)
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return {
                "Description": "Could not generate content for this species.",
                "Habitat": "Unknown",
                "Diet": "Unknown",
                "Fun Fact": "Please try again later."
            }

    def _parse_response(self, text):
        parsed = {
            "Description": "Information not found.",
            "Habitat": "Unknown",
            "Diet": "Unknown",
            "Fun Fact": "Unknown"
        }
        
        lines = text.strip().split('\n')
        current_key = None
        current_val = []

        for line in lines:
            line = line.strip()
            
            # Clean Markdown formatting like **key:**
            clean_line = line.replace("*", "").replace("#", "").replace("_", "").strip()
            
            if not clean_line:
                continue
            
            # Check if line starts with a key
            found_key = False
            for key in parsed.keys():
                if clean_line.startswith(key + ":"):
                    if current_key:
                        parsed[current_key] = " ".join(current_val).strip()
                    current_key = key
                    current_val = [clean_line[len(key)+1:].strip()]
                    found_key = True
                    break
            
            if not found_key and current_key:
                current_val.append(clean_line)
                
        if current_key and current_val:
            parsed[current_key] = " ".join(current_val).strip()
            
        return parsed
