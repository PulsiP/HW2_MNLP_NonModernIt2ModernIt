from google import genai
class Genai:
    """ 
    This class allows interaction with the Gemini API.
    It can be used to get a response from a text input using a specified model.
    """
    def __init__(self, api_key, model_name):
        """
        Initializes the Genai class instance.

        Args:
            api_key (str): Your API key for authenticating with the Gemini service.
            model_name (str): The name of the model to use.
        """
        self.api_key = api_key              
        self.model_name = model_name       
        self.client = genai.Client(api_key=api_key) 

    def get_response(self, text_content):
        """
        Sends a text prompt to the model and returns the generated response.

        Args:
            text_content (str or list): The input text or prompt to send to the model.

        Returns:
            object: The response object generated by the model.
        """
        response = self.client.models.generate_content(
            model=self.model_name,       
            contents=text_content        
        )
        return response 


if __name__ == "__main__":
    genai = Genai("AIzaSyADg7nkKm1byyBq09MY4CcxZ2dxxzJyPWI", "gemini-2.0-flash")

    r = genai.get_response("ciao")

    print(r.text)
