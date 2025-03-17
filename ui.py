import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import numpy as np
import os
import time
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio,structural_similarity


def encrypt_doc(doc_path, key, output_encrypted_path):
    with open(doc_path, "rb") as file:
        pdf_data = file.read()
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.iv
    encrypted_data = cipher.encrypt(pad(pdf_data, AES.block_size))

    with open(output_encrypted_path, "wb") as file:
        file.write(iv + encrypted_data)

    print(f"Encrypted Doc saved to '{output_encrypted_path}'")

    return iv + encrypted_data  # Concatenate IV with encrypted data for decryption

# Function to embed data into an image using LSB steganography
def embed_data_to_image(image_file, output_image_path, data):
    # Open the image directly from the file object
    image = Image.open(image_file)
    image_data = np.array(image)

    # Flatten the image data and prepare to embed
    flat_image_data = image_data.flatten()
    data_bits = ''.join(format(byte, '08b') for byte in data)
    data_length = len(data_bits)

    # Embed data length (32 bits for the size of the data in bits)
    length_bits = format(data_length, '032b')
    total_bits = length_bits + data_bits

    if len(total_bits) > len(flat_image_data):
        raise ValueError("Data size exceeds image capacity.")

    # Embed data into the LSB of the image
    for i in range(len(total_bits)):
        flat_image_data[i] = (flat_image_data[i] & ~1) | int(total_bits[i])

    # Reshape and save the image
    embedded_image_data = flat_image_data.reshape(image_data.shape)
    embedded_image = Image.fromarray(embedded_image_data)
    embedded_image.save(output_image_path)



    # Function to extract data from an image using LSB steganography
def extract_data_from_image(image_path):
    start_time_ilsb = time.time()
    image = Image.open(image_path)
    image_data = np.array(image)

    # Flatten the image data for easier manipulation
    flat_image_data = image_data.flatten()

    # Extract the first 32 bits to determine the data length
    length_bits = ''.join(str(flat_image_data[i] & 1) for i in range(32))
    data_length = int(length_bits, 2)

    # Extract the encrypted data based on the extracted length
    data_bits = ''.join(str(flat_image_data[i] & 1) for i in range(32, 32 + data_length))
    data_bytes = bytes(int(data_bits[i:i+8], 2) for i in range(0, len(data_bits), 8))

    end_time_ilsb = time.time()
    ilsb_execution_time = end_time_ilsb - start_time_ilsb

    # Clean the image by resetting the LSBs to 0
    for i in range(32 + data_length):
        flat_image_data[i] = flat_image_data[i] & ~1  # Reset LSB to 0

    # Reshape the image data back to its original shape
    cleaned_image_data = flat_image_data.reshape(image_data.shape)
    cleaned_image = Image.fromarray(cleaned_image_data)

    return data_bytes, cleaned_image, ilsb_execution_time

# Function to decrypt the PDF or DOCX document
def decrypt_doc(encrypted_data, key):
    iv = encrypted_data[:16]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(encrypted_data[16:]), AES.block_size)
    return decrypted_data


st.set_page_config(
    page_title="NEA",
)

# navigation/option
with st.sidebar:
   selected = option_menu(
        menu_title="Main Menu",  
        options=["Home","Insertion","Extraction"], 
        icons=["house", "record-circle"],  
        menu_icon="cast",  # optional
        default_index=0,  # optional         
)

if selected == "Home":
    st.write("# NEA ")
    st.write(
    """
    **Protect Your Document Here**.
    """
    )
    
    st.caption("Created by **Zein Rivo**")

if selected == "Insertion":

    uploaded_file = st.file_uploader("Upload a PDF or DOCX document", type=["pdf", "docx"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        st.write(f"Uploaded document: {uploaded_file.name}")

        if uploaded_file.type == "application/pdf":
            st.write("This is a PDF document.")
        elif uploaded_file.type == "application/msword" or uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            st.write("This is a DOCX document.")
    else:
        st.info("Please upload a document.")


    uploaded_image = st.file_uploader("Upload an image (1280x720 resolution, PNG format)", type=["png"])


    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        if image.size == (1280, 720):
            if image.format == "PNG":
                st.success("Image uploaded successfully!")
            else:
                st.error("Invalid image format. Please upload a PNG image.")
        else:
            st.error(f"Invalid image resolution. Please upload an image with 1280x720 resolution. Your image resolution is {image.size}.")
    else:
        st.info("Please upload an image.")

    if uploaded_image is not None:  
        st.write("---")  
        
        check_option = st.radio("Do you want to process the uploaded image with ESRGAN?", ("Yes", "No"),index=None)
        
        if check_option is not None:
        
            if check_option == "Yes":
                st.write("1")  
                st.markdown("[Click here to process](https://colab.research.google.com/drive/1RdYkUjNjbMSfO19WnU6Yp8_L1eGLncfO?usp=drive_link)")

                ae_download_status = st.radio("Have you downloaded the processed image?", ("Yes", "No"),index=None)

                if ae_download_status == "Yes":
                    st.write("1")
                    ae_image = st.file_uploader("Upload the processed image", type=["png"])
                    
                    st.write("Please enter a 16-character encryption key:")
                    key = st.text_input("Encryption Key", max_chars=16, placeholder="Enter exactly 16 characters")
                    if key:
                        if len(key) == 16:
                            st.success("Thank You.")
                            st.write("1") 

                            if uploaded_file.type == "application/pdf":
                                file_path = "temp_uploaded.pdf"
                            elif uploaded_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                                file_path = "temp_uploaded.docx"

                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            encrypted_file_path = "encrypted_file.bin"

                            start_time_aes = time.time()
                            encrypted_data = encrypt_doc(file_path, key.encode(), encrypted_file_path)
                            end_time_aes = time.time()
                            aes_execution_time = end_time_aes - start_time_aes


                            # Embed the encrypted data into the image
                            output_image_path = "stego_image.png"
                            start_time_lsb = time.time()
                            embed_data_to_image(ae_image, output_image_path, encrypted_data)
                            end_time_lsb = time.time()
                            lsb_execution_time = end_time_lsb - start_time_lsb


                            st.write(f"AES-128 Encryption Time: {aes_execution_time:.4f} seconds")
                            st.write(f"LSB Embedding Time: {lsb_execution_time:.4f} seconds")


                            # Provide download link for the stego image
                            with open(output_image_path, "rb") as f:
                                st.download_button(
                                    label="Download Stego Image",
                                    data=f,
                                    file_name="stego_image.png",
                                    mime="image/png"
                                )

                            st.write("---")
                            st.write("### Image Quality Metrics")

                            # Load the cover image (original uploaded image)
                            cover_image = Image.open(ae_image)
                            cover_image_array = np.array(cover_image)

                            # Load the stego image (embedded image)
                            stego_image_array = np.array(Image.open(output_image_path))

                            # Calculate MSE and PSNR
                            mse_value = mean_squared_error(cover_image_array, stego_image_array)
                            psnr_value = peak_signal_noise_ratio(cover_image_array, stego_image_array)
                            ssim_value = structural_similarity(cover_image_array, stego_image_array, multichannel=True)

                            # Display the results
                            st.write(f"Mean Squared Error (MSE): {mse_value:.4f}")
                            st.write(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.4f} dB")
                            st.write(f"Structural Similarity Index (SSIM): {ssim_value:.4f}")


                            # Clean up temporary files
                            os.remove(file_path)
                            os.remove(encrypted_file_path)
                            os.remove(output_image_path)

                        else:
                            st.error("The key must be exactly 16 characters long.")
                            st.write("0") 
                    else:
                        st.info("Waiting for input...")




                elif ae_download_status == "No":
                    st.write("You must download the processed image.")


            elif check_option == "No":
                st.write("0")  
                st.write("Please enter a 16-character encryption key:")
                key = st.text_input("Encryption Key", max_chars=16, placeholder="Enter exactly 16 characters")
                if key:
                        if len(key) == 16:
                            st.success("Thank You.")
                            st.write("1") 

                            if uploaded_file.type == "application/pdf":
                                file_path = "temp_uploaded.pdf"
                            elif uploaded_file.type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                                file_path = "temp_uploaded.docx"

                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            encrypted_file_path = "encrypted_file.bin"

                            start_time_aes = time.time()
                            encrypted_data = encrypt_doc(file_path, key.encode(), encrypted_file_path)
                            end_time_aes = time.time()
                            aes_execution_time = end_time_aes - start_time_aes


                            # Embed the encrypted data into the image
                            output_image_path = "stego_image.png"
                            start_time_lsb = time.time()
                            embed_data_to_image(uploaded_image, output_image_path, encrypted_data)
                            end_time_lsb = time.time()
                            lsb_execution_time = end_time_lsb - start_time_lsb


                            st.write(f"AES-128 Encryption Time: {aes_execution_time:.4f} seconds")
                            st.write(f"LSB Embedding Time: {lsb_execution_time:.4f} seconds")


                            # Provide download link for the stego image
                            with open(output_image_path, "rb") as f:
                                st.download_button(
                                    label="Download Stego Image",
                                    data=f,
                                    file_name="stego_image.png",
                                    mime="image/png"
                                )

                            st.write("---")
                            st.write("### Image Quality Metrics")

                            # Load the cover image (original uploaded image)
                            cover_image = Image.open(uploaded_image)
                            cover_image_array = np.array(cover_image)

                            # Load the stego image (embedded image)
                            stego_image_array = np.array(Image.open(output_image_path))

                            # Calculate MSE and PSNR
                            mse_value = mean_squared_error(cover_image_array, stego_image_array)
                            psnr_value = peak_signal_noise_ratio(cover_image_array, stego_image_array)
                            ssim_value = structural_similarity(cover_image_array, stego_image_array, multichannel=True)
                            # Display the results
                            st.write(f"Mean Squared Error (MSE): {mse_value:.4f}")
                            st.write(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.4f} dB")
                            st.write(f"Structural Similarity Index (SSIM): {ssim_value:.4f}")   

                            # Clean up temporary files
                            os.remove(file_path)
                            os.remove(encrypted_file_path)
                            os.remove(output_image_path)

                        else:
                            st.error("The key must be exactly 16 characters long.")
                            st.write("0") 
                else:
                    st.info("Waiting for input...")

        else:
            st.info("Please choose whether to process the image with ESRGAN.")




if selected == "Extraction":

    stego_image = st.file_uploader("Upload the stego image", type=["png"])
    
    if stego_image is not None:
        st.success("Stego image uploaded successfully!")
        st.write(f"Uploaded stego image: {stego_image.name}")

        # Input the encryption key
        key = st.text_input("Enter the 16-character encryption key", max_chars=16, placeholder="Enter exactly 16 characters")

        if key:
            if len(key) == 16:
                key = key.encode()

                # Extract the encrypted data and clean the image

                encrypted_data, cleaned_image, ilsb_ex_time = extract_data_from_image(stego_image)

                # Attempt to decrypt the data with the provided key
                try:
                    
                    start_time_iaes = time.time()
                    decrypted_data = decrypt_doc(encrypted_data, key)
                    end_time_iaes = time.time()
                    iaes_execution_time = end_time_iaes - start_time_iaes

                    # If decryption succeeds, the key is correct
                    st.success("Key is correct! Decryption successful.")

                    # Save the cleaned image to a temporary file
                    cleaned_image_path = "cleaned_image.png"
                    cleaned_image.save(cleaned_image_path)

                    # Provide download link for the cleaned image
                    with open(cleaned_image_path, "rb") as f:
                        st.download_button(
                            label="Download Cleaned Image",
                            data=f,
                            file_name="cleaned_image.png",
                            mime="image/png"
                        )

                    # Determine the file type based on the decrypted data
                    if decrypted_data.startswith(b'%PDF'):
                        file_extension = "pdf"
                        mime_type = "application/pdf"
                    else:
                        file_extension = "docx"
                        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

                    st.write(f"Inverse LSB Time: {ilsb_ex_time:.4f} seconds")
                    st.write(f"Inverse AES-128 Time: {iaes_execution_time:.4f} seconds")
                    
                    
                    # Provide download link for the decrypted document
                    st.download_button(
                        label=f"Download Decrypted Document ({file_extension.upper()})",
                        data=decrypted_data,
                        file_name=f"decrypted_document.{file_extension}",
                        mime=mime_type
                    )

                    # Clean up the temporary cleaned image file
                    os.remove(cleaned_image_path)

                except (ValueError, KeyError) as e:
                    st.error("Decryption failed. The key is incorrect or the data is corrupted.")
                    st.info("Please enter the correct key.")

            else:
                st.error("The key must be exactly 16 characters long.")
        else:
            st.info("Please enter the encryption key.")
    else:
        st.info("Please upload the stego image.")
