import streamlit as st
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import requests
from byaldi import RAGMultiModalModel
from io import BytesIO
import torch
import re
import base64

# INTIALIZING RAGMultiModalModel
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali", verbose=10)
model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# CREATING IMAGE INDEX 
def create_RAG_index(image_path):
    RAG.index(
        input_path=image_path,
        index_name="image_index",
        store_collection_with_index=True,
        overwrite=True,
    )

# EXTRACTING TEXT
def extract_relevant_text(output):
    text = output[0]
    lines = text.split('\n')
    relevant_text = []

    for line in lines:
        
        if re.match(r'[A-Za-z0-9]', line):  
            relevant_text.append(line.strip())

    return "\n".join(relevant_text)

# FORMING TEXT QUERY
def ocr_image(image_path,text_query):
    if text_query:
      create_rag_index(image_path)
      results = RAG.search(text_query, k=1, return_base64_results=True)

      image_data = base64.b64decode(results[0].base64)
      image = Image.open(BytesIO(image_data))
    else:
      image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "explain all text find in the image."
                }
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    inputs = inputs.to(device)

    output_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

   # EXTRACTING TEXT
    relevant_text = extract_relevant_text(output_text)

    return relevant_text

# SEARCHING THE REQUIRED TEXT 
def ocr_and_search(image, keyword):
    extracted_text = ocr_image(image,keyword)
    #print(extracted_text)
    if keyword =='':
      return extracted_text , 'Please Enter a Keyword'

    else:
      print("Not Found")
    return extracted_text 

# CREATING THE INTERFACE
st.title("TEXT EXTRACTION AND SEARCH USING OCR")
st.write("Upload an image to extract text in Hindi and English and search for keywords.")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
keyword = st.text_input("Enter Keyword")

if st.button("Submit"):
    if uploaded_image is not None:
        image_path = uploaded_image.name
        extracted_text, highlighted_text = ocr_and_search(image_path, keyword)
        st.write("Extracted Text:")
        st.write(extracted_text)
        st.write("Search Result:")
        st.write(highlighted_text, unsafe_allow_html=True)
    else:
        st.write("Please upload an image")