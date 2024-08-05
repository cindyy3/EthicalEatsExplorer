import gradio as gr
from sentence_transformers import SentenceTransformer, util
import openai
import random
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize paths and model identifiers for easy configuration and maintenance
filename = "output_topic_details.txt"  # Path to the file storing restaurant-specific details
retrieval_model_name = 'output/sentence-transformer-finetuned/'

# Initialize the system message for the chatbot
system_message = "You are a restaurant recommending chatbot that suggests ONLY ONE restaurant in Seattle from the restaurant database based on the criteria the user provides."

# Initial system message to set the behavior of the assistant
messages = [{"role": "system", "content": system_message}]

# Load the SentenceTransformer model
try:
    retrieval_model = SentenceTransformer(retrieval_model_name)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Failed to load models: {e}")

def load_and_preprocess_text(filename):
    """
    Load and preprocess text from a file, removing empty lines and stripping whitespace.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            segments = [line.strip() for line in file if line.strip()]
        print("Text loaded and preprocessed successfully.")
        return segments
    except Exception as e:
        print(f"Failed to load or preprocess text: {e}")
        return []

segments = load_and_preprocess_text(filename)

def find_relevant_segment(user_query, segments):
    """
    Find the most relevant text segment for a user's query using cosine similarity among sentence embeddings.
    This version finds the best match based on the content of the query.
    """
    try:
        lower_query = user_query.lower()
        query_embedding = retrieval_model.encode(lower_query)
        segment_embeddings = retrieval_model.encode(segments)
        similarities = util.pytorch_cos_sim(query_embedding, segment_embeddings)[0]
        best_idx = similarities.argmax()
        return segments[best_idx]
    except Exception as e:
        print(f"Error in finding relevant segment: {e}")
        return ""

def generate_response(user_query, relevant_segment):
    """
    Generate a response emphasizing the bot's capability in suggesting a restaurant.
    """
    try:
        user_message = f"Here is a local restaurant based on your information: {relevant_segment}"
        messages.append({"role": "user", "content": user_message})
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.2,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        output_text = response['choices'][0]['message']['content'].strip()
        messages.append({"role": "assistant", "content": output_text})
        return output_text
    except Exception as e:
        print(f"Error in generating response: {e}")
        return f"Error in generating response: {e}"

# Define a sample list of restaurants (replace this with your actual data source)
restaurants = [
    {
    "name": "Harvest Beat",
        "price": "High",
        "cuisine": "American",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.harvestbeat.com/"
    },
    {
        "name": "Plum Bistro",
        "price": "Moderate",
        "cuisine": "American",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://plumbistro.com/"
    },
    {
        "name": "Portage Bay Cafe",
        "price": "Low",
        "cuisine": "American",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.portagebaycafe.com/"
    },
    {
        "name": "Duke's Seafood",
        "price": "High",
        "cuisine": "American",
        "gluten_free": True,
        "vegan": False,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://www.dukesseafood.com/"
    },
    {
        "name": "Olmstead",
        "price": "Moderate",
        "cuisine": "American",
        "gluten_free": False,
        "vegan": False,
        "lactose_intolerant": False,
        "pescatarian": False,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://www.olmsteadseattle.com/home"
    },
    {
        "name": "Veggie Grill",
        "price": "Low",
        "cuisine": "American",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.veggiegrill.com/menus/"
    },
    {
        "name": "Cafe Flora",
        "price": "Moderate",
        "cuisine": "American",
        "gluten_free": False,
        "vegan": True,
        "lactose_intolerant": False,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://florarestaurantgroup.com/"
    },
    {
        "name": "Bounty Kitchen",
        "price": "Low",
        "cuisine": "American",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": False,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "http://www.bountykitchenseattle.com/"
    },
    {
        "name": "No Bones Beach Club",
        "price": "Low",
        "cuisine": "American",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": False,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://nobonesbeachclub.com/menu/"
    },
    {
        "name": "Chiho Bistro",
        "price": "Low",
        "cuisine": "Chinese",
        "gluten_free": False,
        "vegan": True,
        "lactose_intolerant": False,
        "pescatarian": False,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://www.chihobistro.com/"
    },
    {
        "name": "Uptown China",
        "price": "Moderate",
        "cuisine": "Chinese",
        "gluten_free": False,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://uptown-china.com/"
    },
    {
        "name": "Meskel Ethiopian Restaurant",
        "price": "Moderate",
        "cuisine": "Ethiopian",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://meskelethiopian.com/menu"
    },
    {
        "name": "Habesha Cafe",
        "price": "Moderate",
        "cuisine": "Ethiopian",
        "gluten_free": False,
        "vegan": True,
        "lactose_intolerant": False,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://habesha.cafe/"
    },
    {
        "name": "Musang",
        "price": "High",
        "cuisine": "Filipino",
        "gluten_free": True,
        "vegan": False,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://www.musangseattle.com/"
    },
    {
        "name": "Musang Seattle",
        "price": "Moderate",
        "cuisine": "Filipino",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.musangseattle.com/musang"
    },
    {
        "name": "Gold Coast Ghal",
        "price": "Moderate",
        "cuisine": "Ghanaian",
        "gluten_free": False,
        "vegan": False,
        "lactose_intolerant": False,
        "pescatarian": False,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://www.goldcoastghal.com/"
    },
    {
        "name": "Marination Ma Kai",
        "price": "Low",
        "cuisine": "Hawaiian-Korean",
        "gluten_free": True,
        "vegan": False,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://marinationmobile.com/menu"
    },
    {
        "name": "Pabla Indian Cuisine",
        "price": "Low",
        "cuisine": "Indian",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": True,
        "kosher": False,
        "vegetarian": True,
        "website": "http://www.pablacuisine"
    },
    {
        "name": "Mint Progressive Indian",
        "price": "Moderate",
        "cuisine": "Indian",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.mintprogressive.com/"
    },
    {
        "name": "Cantinetta",
        "price": "Moderate",
        "cuisine": "Italian",
        "gluten_free": False,
        "vegan": False,
        "lactose_intolerant": False,
        "pescatarian": False,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://www.cantinettausa.com/wallingfordYes"
    },
    {
        "name": "The Pink Door",
        "price": "High",
        "cuisine": "Italian",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": False,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.thepinkdoor.net/"
    },
    {
        "name": "Mashiko",
        "price": "High",
        "cuisine": "Japanese",
        "gluten_free": False,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://www.mashikorestaurant.com/"
    },
    {
        "name": "Ben's Fast Food",
        "price": "Low",
        "cuisine": "Mexican",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://bensfastfood.com/"
    },
    {
        "name": "Sal y Limon",
        "price": "Moderate",
        "cuisine": "Mexican",
        "gluten_free": False,
        "vegan": False,
        "lactose_intolerant": False,
        "pescatarian": False,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.salylimonseattle.com/"
    },
    {
        "name": "El Borracho",
        "price": "Low",
        "cuisine": "Mexican",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.elborracho.co/"
    },
    {
        "name": "Tanoor",
        "price": "Moderate",
        "cuisine": "Middle Eastern",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": True,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.tanoor.com/"
    },
    {
        "name": "Mamnoon",
        "price": "Moderate",
        "cuisine": "Middle Eastern",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://nadimama.com/mamnoon"
    },
    {
        "name": "Gorgeous George's",
        "price": "Moderate",
        "cuisine": "Middle Eastern",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.gorgeousgeorges.com/"
    },
    {
        "name": "Zaina",
        "price": "Low",
        "cuisine": "Middle Eastern",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": True,
        "halal": True,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.yelp.com/biz/zaina-food-drinks-and-friends-seattle-3"
    },
    {
        "name": "Gold Schnitzel",
        "price": "Moderate",
        "cuisine": "Middle Eastern",
        "gluten_free": False,
        "vegan": False,
        "lactose_intolerant": True,
        "pescatarian": False,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://goldschnitzel.com/"
    },
    {
        "name": "Maza Grill",
        "price": "Moderate",
        "cuisine": "Pakistani",
        "gluten_free": False,
        "vegan": False,
        "lactose_intolerant": False,
        "pescatarian": False,
        "allergen_friendly": True,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://mazagrill.co/"
    },
    {
        "name": "Terra Plata",
        "price": "High",
        "cuisine": "Spanish",
        "gluten_free": False,
        "vegan": False,
        "lactose_intolerant": False,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://www.terraplata.com/"
    },
    {
        "name": "Pestle Rock",
        "price": "Moderate",
        "cuisine": "Thai",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://pestlerock.com/"
    },
    {
        "name": "Araya's Place",
        "price": "Low",
        "cuisine": "Thai",
        "gluten_free": False,
        "vegan": True,
        "lactose_intolerant": False,
        "pescatarian": False,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.arayasplace.com/"
    },
    {
        "name": "Cafe Turko",
        "price": "Moderate",
        "cuisine": "Turkish",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": True,
        "kosher": False,
        "vegetarian": True,
        "website": "https://cafeturko.com/#"
    },
    {
        "name": "Blossom Vegetarian",
        "price": "Low",
        "cuisine": "Vietnamese",
        "gluten_free": True,
        "vegan": True,
        "lactose_intolerant": True,
        "pescatarian": False,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": True,
        "website": "https://www.blossomrenton.com/"
    },
    {
        "name": "Tilikum",
        "price": "Moderate",
        "cuisine": "European",
        "gluten_free": False,
        "vegan": False,
        "lactose_intolerant": False,
        "pescatarian": True,
        "allergen_friendly": False,
        "halal": False,
        "kosher": False,
        "vegetarian": False,
        "website": "https://www.tilikumplacecafe.com/"
    }
    # Add more restaurant entries as needed
]

def find_restaurants(criteria):
    """
    Finds restaurants based on the given criteria.
    Parameters:
        criteria (dict): Dictionary containing filtering criteria.
    Returns:
        List of restaurants that match the criteria.
    """
    matching_restaurants = []
    for restaurant in restaurants:
        match = True
        for key, value in criteria.items():
            if key in restaurant:
                if isinstance(restaurant[key], bool):
                    if restaurant[key] != value:
                        match = False
                        break
                elif restaurant[key].lower() != value.lower():
                    match = False
                    break
        if match:
            matching_restaurants.append(restaurant)
    return matching_restaurants

def generate_recommendation(criteria):
    """
    Generates a recommendation based on the criteria.
    Parameters:
        criteria (dict): Dictionary containing filtering criteria.
    Returns:
        String with the recommendation or a message if no matches are found.
    """
    results = find_restaurants(criteria)
    if results:
        recommendations = []
        for result in results:
            recommendation = (
                f"Based on your criteria, I recommend {result['name']}. "
                f"It's a {result['price'].lower()} priced {result['cuisine'].lower()} restaurant with "
                f"{'gluten-free options' if result['gluten_free'] else 'no gluten-free options'}, "
                f"{'vegan options' if result['vegan'] else 'no vegan options'}, "
                f"{'lactose-intolerant options' if result['lactose_intolerant'] else 'no lactose-intolerant options'}, "
                f"{'pescatarian options' if result['pescatarian'] else 'no pescatarian options'}, "
                f"{'allergen-friendly options' if result['allergen_friendly'] else 'no allergen-friendly options'}, "
                f"{'halal options' if result['halal'] else 'no halal options'}, "
                f"{'kosher options' if result['kosher'] else 'no kosher options'}, "
                f"and { 'vegetarian options' if result['vegetarian'] else 'no vegetarian options'}. "
                f"Visit their website for more details: {result['website']}"
            )
            recommendations.append(recommendation)
        random_index = random.randint(0, len(recommendations) - 1)
        return recommendations[random_index]
    else:
        return "Sorry, no restaurants meet your criteria. Please try adjusting your filters."

def query_model(question):
    """
    Process a question, find relevant information, and generate a response.
    """
    if question == "":
        return "Give me your preferences..."
    
    if "restaurant" in question.lower():
        # Extract criteria from the question
        criteria = {}
        if "gluten-free" in question.lower():
            criteria["gluten_free"] = True
        if "vegan" in question.lower():
            criteria["vegan"] = True
        if "lactose-intolerant" in question.lower():
            criteria["lactose_intolerant"] = True
        if "pescatarian" in question.lower():
            criteria["pescatarian"] = True
        if "allergen-friendly" in question.lower():
            criteria["allergen_friendly"] = True
        if "halal" in question.lower():
            criteria["halal"] = True
        if "kosher" in question.lower():
            criteria["kosher"] = True
        if "vegetarian" in question.lower():
            criteria["vegetarian"] = True
        
        # Extract price and cuisine
        if "low" in question.lower():
            criteria["price"] = "Low"
        elif "moderate" in question.lower():
            criteria["price"] = "Moderate"
        elif "high" in question.lower():
            criteria["price"] = "High"
        
        if any(cuisine in question.lower() for cuisine in ["american", "indian", "middle eastern", "chinese", "italian", "thai", "hawaiian-korean", "japanese", "ethiopian", "pakistani", "mexican", "ghanaian", "vietnamese", "filipino", "spanish", "turkish"]):
            criteria["cuisine"] = next(cuisine for cuisine in ["american", "indian", "middle eastern", "chinese", "italian", "thai", "hawaiian-korean", "japanese", "ethiopian", "pakistani", "mexican", "ghanaian", "vietnamese", "filipino", "spanish", "turkish"] if cuisine in question.lower())
        
        response = generate_recommendation(criteria)
    else:
        relevant_segment = find_relevant_segment(question, segments)
        if not relevant_segment:
            return "Could not find specific information. Please refine your question."
        response = generate_response(question, relevant_segment)
    
    return response

# Define the welcome message and specific topics the chatbot can provide information about
welcome_message = """
# Welcome to Ethical Eats Explorer!
## Your AI-driven assistant for restaurant recs in Seattle. Created by Saranya, Cindy, and Liana of the 2024 Kode With Klossy Seattle Camp.
"""
topics = """
### Please give me your restaurant preferences:
- Dietary Restrictions
- Cuisine Preferences (optional)
    - Cuisines: American, Indian, Middle Eastern, Chinese, Italian, Thai, Hawaiian-Korean, Japanese, Ethiopian, Pakistani, Mexican, Ghanaian, Vietnamese, Filipino, Spanish, Turkish
- Budget Preferences (Low: $0 - $20, Moderate: $20 - $30, High: $30+ - per person)
Please send your message in the format: "Could you give me a (cuisine) restaurant with (dietary restriction) options that is (budget) budget?"
"""
# Setup the Gradio Blocks interface with custom layout components
with gr.Blocks(theme='JohnSmith9982/small_and_pretty') as demo:
    gr.Image("eeepic.png", show_label = False, show_share_button = False, show_download_button = False)
    gr.Markdown(welcome_message)  # Display the formatted welcome message
    with gr.Row():
        with gr.Column():
            gr.Markdown(topics)  # Show the topics on the left side
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Your question", placeholder="Give me your information...")
            answer = gr.Textbox(label="Explorer's Response", placeholder="Explorer will respond here...", interactive=False, lines=10)
            submit_button = gr.Button("Submit")
            submit_button.click(fn=query_model, inputs=question, outputs=answer)
# Launch the Gradio app to allow user interaction
demo.launch(share=True)