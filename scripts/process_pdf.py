#!/usr/bin/env python3
"""
Simple PDF Processing Script
Reads configuration from YAML file
"""

import PyPDF2
import os
import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {str(e)}")
        return None


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        return None
    
    print(f"📄 Reading PDF: {pdf_path}")
    
    try:
        # Open PDF file
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            print(f"📊 Total pages: {num_pages}")
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        return full_text
    
    except Exception as e:
        print(f"❌ Error reading PDF: {str(e)}")
        return None


def save_text_to_file(text, output_path):
    """Save extracted text to a file"""
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✅ Text saved to: {output_path}")
        print(f"📝 Total characters: {len(text)}")
        return True
    except Exception as e:
        print(f"❌ Error saving text: {str(e)}")
        return False


def main():
    """Main function"""
    
    print("="*50)
    print("PDF Text Extraction")
    print("="*50)
    
    # Load configuration
    config = load_config("config.yaml")
    
    if not config or 'pdf' not in config:
        print("❌ Invalid configuration!")
        return
    
    pdf_path = config['pdf']['input_path']
    output_path = config['pdf']['output_path']
    
    print(f"Input PDF: {pdf_path}")
    print(f"Output TXT: {output_path}")
    print("="*50)
    
    # Extract text
    extracted_text = extract_text_from_pdf(pdf_path)
    
    if extracted_text:
        # Save to file
        save_text_to_file(extracted_text, output_path)
        print("\n✅ Processing complete!")
    else:
        print("\n❌ Processing failed!")


if __name__ == "__main__":
    main()
