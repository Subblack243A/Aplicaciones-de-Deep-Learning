import os
import concurrent.futures
import argostranslate.package
import argostranslate.translate
from tqdm import tqdm

class TranslationService:
    """
    Service responsible for handling text translation using argostranslate.
    Handles only translation logic and package management.
    Can be extended for other translation libraries if needed without modifying client code.
    """

    def __init__(self, from_code: str = "en", to_code: str = "es"):
        """
        Initializes the translation service and ensures the required language package is installed.
        
        Args:
            from_code (str): Source language code (default: "en").
            to_code (str): Target language code (default: "es").
        """
        self.from_code = from_code
        self.to_code = to_code
        self._ensure_package_installed()
        
        # Cache the translation model to avoid reloading it for every request
        self.translation_model = argostranslate.translate.get_translation_from_codes(from_code, to_code)

    def _ensure_package_installed(self):
        """
        Private method to check and install the required translation package.
        Optimized to check local installations first before reaching out to the network.
        """
        print(f"Checking for installed packages: {self.from_code} -> {self.to_code}...")
        
        # Check if the package is already installed
        installed_packages = argostranslate.package.get_installed_packages()
        is_installed = any(
            pkg.from_code == self.from_code and pkg.to_code == self.to_code 
            for pkg in installed_packages
        )

        if is_installed:
            print("Package already installed. Skipping download.")
            return

        print("Package not found locally. Updating package index...")
        try:
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                filter(lambda x: x.from_code == self.from_code and x.to_code == self.to_code, available_packages), 
                None
            )

            if package_to_install:
                print(f"Downloading and installing package: {package_to_install}...")
                argostranslate.package.install_from_path(package_to_install.download())
                print("Package installed successfully.")
            else:
                print("No specific package found in index.")
        except Exception as e:
            print(f"Warning: Could not update/install packages automatically. Error: {e}")

    def translate_text(self, text: str) -> str:
        """
        Translates a single string of text using the cached model.
        """
        try:
            return self.translation_model.translate(text)
        except Exception as e:
            print(f"Error translating text chunk: {e}")
            return text

    def translate_file(self, input_file_path: str, output_file_path: str, batch_size: int = 50, max_workers: int = 4):
        """
        Reads a file, translates its content in batches (for speed), and saves the result.
        Uses parallel processing to speed up translation.
        
        Args:
            input_file_path (str): Path to the source text file.
            output_file_path (str): Path where the translated text will be saved.
            batch_size (int): Number of lines to translate in one go.
            max_workers (int): Number of threads to use for parallel processing.
        """
        if not os.path.exists(input_file_path):
            print(f"Error: Input file '{input_file_path}' does not exist.")
            return

        print(f"Reading file '{input_file_path}'...")
        with open(input_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            print("The input file is empty.")
            return

        print(f"Translating {len(lines)} lines (Batch size: {batch_size}, Workers: {max_workers})...")
        
        # Prepare batches
        batches = []
        for i in range(0, len(lines), batch_size):
            batches.append(lines[i : i + batch_size])

        # Helper function to process a batch
        def process_batch(batch):
            batch_text = "".join(batch)
            if batch_text.strip():
                return self.translate_text(batch_text)
            else:
                # Keep original formatting for empty lines/batches
                return batch_text 

        # Process batches in parallel
        # We use thread pool as CTranslate2 (underlying argostranslate) releases GIL often
        translated_batches = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map maintains order of results matching order of inputs
            results = executor.map(process_batch, batches)
            
            # Wrap in list and tqdm for progress bar
            translated_batches = list(tqdm(results, total=len(batches), desc="Processing Batches", unit="batch"))

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                for block in translated_batches:
                    f.write(block)
                    
            print(f"Success! Translation saved to '{output_file_path}'.")
        except Exception as e:
            print(f"Error saving translated file: {e}")

if __name__ == "__main__":
    
    # Example usage
    INPUT_FILE = "Snuff - Slipknot Lyrics.txt"
    OUTPUT_FILE = f"{INPUT_FILE}_letra_espanol.txt"   
    
    translator = TranslationService(from_code="en", to_code="es")
    translator.translate_file(INPUT_FILE, OUTPUT_FILE)