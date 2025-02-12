import os
import pytesseract
from pdf2image import convert_from_path
from docx import Document
from PIL import Image
import pandas as pd
import logging
from tqdm import tqdm

class CVTextExtractor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Setup logging
        logging.basicConfig(
            filename=os.path.join(output_folder, 'extraction_log.txt'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def extract_from_pdf(self, pdf_path):
        try:
            images = convert_from_path(pdf_path)
            text = ""

            for image in images:
                text += pytesseract.image_to_string(image)

            return text.strip()
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None

    def extract_from_docx(self, docx_path):
        try:
            doc = Document(docx_path)
            text = ""

            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            return text.strip()
        except Exception as e:
            logging.error(f"Error processing DOCX {docx_path}: {str(e)}")
            return None

    def clean_text(self, text):
        if text:
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Remove unnecessary line breaks
            text = text.replace('\n\n', '\n')
            return text
        return ""

    def process_document(self, file_path):
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            return self.extract_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self.extract_from_docx(file_path)
        else:
            logging.warning(f"Unsupported file format: {file_path}")
            return None

    def process_folder(self):
        results = []

        # Get list of files
        files = [f for f in os.listdir(self.input_folder)
                if f.lower().endswith(('.pdf', '.docx', '.doc'))]

        print(f"Found {len(files)} files to process")

        # Process each file
        for file_name in tqdm(files):
            file_path = os.path.join(self.input_folder, file_name)

            logging.info(f"Processing {file_name}")

            try:
                # Extract text
                extracted_text = self.process_document(file_path)

                if extracted_text:
                    # Clean text
                    cleaned_text = self.clean_text(extracted_text)

                    # Save individual text file
                    output_file = os.path.join(
                        self.output_folder,
                        f"{os.path.splitext(file_name)[0]}.txt"
                    )

                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)

                    results.append({
                        'file_name': file_name,
                        'status': 'success',
                        'text': cleaned_text
                    })

                    logging.info(f"Successfully processed {file_name}")
                else:
                    results.append({
                        'file_name': file_name,
                        'status': 'failed',
                        'text': ''
                    })
                    logging.error(f"Failed to extract text from {file_name}")

            except Exception as e:
                logging.error(f"Error processing {file_name}: {str(e)}")
                results.append({
                    'file_name': file_name,
                    'status': 'error',
                    'text': str(e)
                })

        # Create summary DataFrame
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_folder, 'extraction_summary.csv'), index=False)

        return results

input_folder = 'CVs/'  # Adjust path as needed
output_folder = 'CVs/output'  # Adjust path as needed

# Create extractor instance
extractor = CVTextExtractor(input_folder, output_folder)

# Process all documents
results = extractor.process_folder()

# Print summary
success_count = sum(1 for r in results if r['status'] == 'success')
print(f"\nProcessing complete!")
print(f"Successfully processed: {success_count}/{len(results)} files")
print(f"Results saved to: {output_folder}")