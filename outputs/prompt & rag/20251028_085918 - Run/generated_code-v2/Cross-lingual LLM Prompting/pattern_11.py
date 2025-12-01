import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(layout="wide", page_title="Multilingual Customer Support Chatbot")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    return tokenizer, model

tokenizer, model = load_model()

def generate_icl_prompt(customer_query, customer_language_code, target_language_code, product_info_english):
    prompt = f"""You are a helpful multilingual customer support assistant. Your goal is to understand customer queries in various languages, find relevant information (which might be in English), and respond clearly and accurately in the customer's original language. You will be provided with product information in English. Always respond in the customer's language.

Here are some examples of how to answer cross-lingual queries:

---
Customer Query (French): Je n'arrive pas à connecter mon casque Bluetooth. Quelqu'un peut m'aider ?
Product Info (English): Troubleshooting steps for Bluetooth headphones: 1. Ensure headphones are charged. 2. Put headphones in pairing mode. 3. Go to device settings and select 'Add Bluetooth Device'. 4. Select your headphones from the list. If issues persist, try restarting both devices.
CHATBOT RESPONSE (French): Pour connecter votre casque Bluetooth, veuillez suivre ces étapes : 1. Assurez-vous que le casque est chargé. 2. Mettez le casque en mode d'appairage. 3. Accédez aux paramètres de votre appareil et sélectionnez 'Ajouter un appareil Bluetooth'. 4. Sélectionnez votre casque dans la liste. Si les problèmes persistent, essayez de redémarrer les deux appareils.
---
Customer Query (Spanish): ¿Cómo puedo limpiar mi cafetera expreso?
Product Info (English): To clean your espresso machine: 1. Descale regularly with a descaling solution. 2. Wipe the exterior with a damp cloth. 3. Clean the portafilter and brew head after each use. 4. Backflush weekly with a blind filter and coffee detergent.
CHATBOT RESPONSE (Spanish): Para limpiar su cafetera expreso: 1. Descalcifique regularmente con una solución descalcifiante. 2. Limpie el exterior con un paño húmedo. 3. Limpie el portafiltro y el grupo de preparación después de cada uso. 4. Realice un retrolavado semanal con un filtro ciego y detergente para café.
---
Customer Query ({customer_language_code}): {customer_query}
Product Info (English): {product_info_english}
CHATBOT RESPONSE ({customer_language_code}):"""
    return prompt

product_info_db = {
    "smartphone": "The XYZ Smartphone features a 6.5-inch OLED display, A15 Bionic chip, 128GB storage, 5G connectivity, and a dual 12MP camera system. It is water-resistant up to 6 meters for 30 minutes. Battery life typically lasts up to 18 hours of video playback. Supports fast charging (20W adapter sold separately). Available in Midnight, Starlight, and Blue.",
    "smartwatch": "The ABC Smartwatch monitors heart rate, blood oxygen, and sleep. It has GPS for tracking outdoor workouts and is water-resistant up to 50 meters. Battery life is up to 7 days in normal mode. Features include customizable watch faces, notification alerts, and contactless payments.",
    "laptop": "The DEF Laptop comes with an Intel Core i7 processor, 16GB RAM, 512GB SSD, and a 14-inch Full HD display. It weighs 1.3 kg and offers up to 10 hours of battery life. Ports include 2x USB-C (Thunderbolt 4), 1x USB-A 3.2, and HDMI. Ideal for productivity and light creative work."
}

st.title("🌍 Multilingual E-commerce Customer Support Chatbot")
st.markdown("This chatbot demonstrates **InCLT Crosslingual Transfer Prompting**. It answers queries in your language by leveraging product information primarily available in English, using cross-lingual examples to boost performance.")

st.sidebar.header("Chatbot Settings")
selected_product = st.sidebar.selectbox("Select a product for context:", list(product_info_db.keys()))

customer_query = st.text_input(f"Ask your question about the {selected_product} in German (e.g., 'Wie lange hält der Akku des Smartphones?')")

if st.button("Get Response") and customer_query:
    st.subheader("Your Query:")
    st.write(customer_query)

    customer_language_code = "de_DE"  # For this demo, assuming German input
    target_language_code = "en_XX"   # Product info is in English
    product_info_english = product_info_db.get(selected_product, "No specific information available.")

    prompt = generate_icl_prompt(customer_query, customer_language_code, target_language_code, product_info_english)
    
    st.subheader("Generated Prompt (for LLM):")
    st.text_area("", prompt, height=300)

    with st.spinner("Generating response..."):
        tokenizer.src_lang = customer_language_code
        encoded_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.lang_code_to_id[customer_language_code],
            max_new_tokens=200,
            num_beams=5,
            early_stopping=True
        )

        llm_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        # Extract the actual chatbot response from the structured output
        response_marker = f"CHATBOT RESPONSE ({customer_language_code}):"
        if response_marker in llm_output:
            final_response = llm_output.split(response_marker, 1)[1].strip()
        else:
            final_response = llm_output.strip() # Fallback if marker is not found

        st.subheader("Chatbot Response:")
        st.success(final_response)
elif st.button("Get Response") and not customer_query:
    st.warning("Please enter a query.")