# pip install mtranslate



import openai
import mtranslate as tr




def rephrase_in_celebrity_style(prompt, celebrity):
  """
  rephrase the message based on a person 
  :return: the new message generated.
  """

  # Set up the GPT-3 API client
  openai.api_key = "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"


  # Use the GPT-3 model to generate text based on the prompt and desired style
  completions = openai.Completion.create(
    engine="text-curie-001",
    prompt=f"Refomulate the following text in the style of {celebrity} : {prompt}",
    max_tokens=200,
    n=1,
    temperature=0.5,
  )

  # Return the generated text
  return completions.choices[0].text
  
def translate(message, lang):
    """
    translate the text to a certain lang
    :return: the new message generated.
    """
    return tr.translate(message, lang, "auto")

