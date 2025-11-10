import streamlit as st
import pandas as pd
import numpy as np
import time
from utils import get_local_images_path, generate_id, get_lorem
from pathlib import Path
from chroma_client import ChromaBase
import torch
import uuid
from pprint import pprint
import tempfile
from PIL import Image
from io import BytesIO
from s3 import S3

@st.cache_resource
def initialize_chroma(_bucket_client: S3) -> ChromaBase:
    chroma_base = ChromaBase(_bucket_client)
    
    all_names = _bucket_client.get_all_files()
    all_ids = [generate_id(_) for _ in all_names]
    
    new_names, new_ids = chroma_base.keep_new_only(filespath=all_names, ids=all_ids)
    metadatas = [{"path": str(name), "name": name} for name in new_names]
    
    with st.spinner('Vérification de la base vectorielle, merci de patienter...'):
        if new_ids:
            for i, batch_embeddings in enumerate(chroma_base.compute_embeddings(filespath=new_names)):
                start = i * 10
                end = start + len(batch_embeddings)
                
                batch_ids = new_ids[start:end]
                batch_metadatas = metadatas[start:end]
                
                chroma_base.add_to_collection(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                    )
        return chroma_base

def main() -> None:
    # Build Streamlit base page
    st.set_page_config(page_title="Recherche inversée sur des images d'archive")
    st.title("Recherche inversée sur des images d'archive")
    st.write("Cette application lancée par la M2RS permet d'effectuer une recherche inversée sur un échantillon du fond 209SUP du ministère des Affaires Étrangères (3563 photographies).")
    with st.sidebar:
        st.subheader('Accueil')
    
    # Instantiate S3
    s3 = S3()
    
    chroma_base = initialize_chroma(_bucket_client=s3)
    
    uploaded_image = st.file_uploader(label="Merci de déposer une image :", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        _, center, _ = st.columns((1,2,1))
        center.image(
            image=image_bytes,
            caption=uploaded_image.name,
            use_column_width=True
        )
        
        image_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image_path = Path(tmp.name)
            image_pil.save(image_path)
        
        st.subheader('Images similaires :')

        image_to_query = [image_path]
        results = chroma_base.query_image(image_to_query=image_to_query, n_results=12)
        
        cols = st.columns(3)
        for i, metadata in enumerate(results['metadatas'][0]):
            # img_path = metadata['path']
            filename = metadata['name']
            
            img_bytes = s3.download_file(filename=filename, embeddings=False)
            
            with cols[i % 3]:
                st.image(
                    image=img_bytes,
                    use_column_width=True,
                    caption=filename
                    )

        st.subheader('Debug :')
        my_results = {}
        for metadata, distance in zip(dict(results)['metadatas'][0], dict(results)['distances'][0]):
            name = metadata['name']
            my_results[name] = distance

        st.write(dict(my_results))

if __name__ == '__main__':
    main()