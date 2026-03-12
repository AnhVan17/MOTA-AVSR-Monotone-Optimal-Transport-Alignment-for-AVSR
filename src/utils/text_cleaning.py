import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalize Vietnamese text using NFC and cleanup.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text (NFC, lowercase, stripped)
    """
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    # Lowercase & strip
    text = text.lower().strip()
    # Remove extra whitespaces
    text = " ".join(text.split())
    return text

def find_text_in_sample(sample: dict) -> str:
    """
    Robust text extraction from WebDataset sample.
    Prioritizes JSON/transcript fields over raw text files.
    Filters out file paths and invalid content.
    
    Args:
        sample (dict): WebDataset sample dictionary
        
    Returns:
        str: Extracted text or empty string
    """
    # 1. Prioritize JSON or explicit 'transcript' keys
    for key in ['json', 'transcript', 'label']:
        if key in sample:
            content = sample[key]
            if isinstance(content, dict):
                # Search common fields
                for sub_key in ['text', 'transcript', 'content', 'label', 'caption']:
                    if sub_key in content and isinstance(content[sub_key], str):
                        return content[sub_key]
            elif isinstance(content, str):
                return content
            elif isinstance(content, bytes):
                try:
                    return content.decode('utf-8').strip()
                except:
                    pass

    # 2. Fallback to 'txt' or 'text' keys (less reliable)
    for key in ['txt', 'text']:
        if key in sample:
            content = sample[key]
            text_str = ""
            
            if isinstance(content, bytes):
                try:
                    text_str = content.decode('utf-8').strip()
                except:
                    continue
            elif isinstance(content, str):
                text_str = content.strip()

            if not text_str: continue
            
            # 3. VALIDATION: Filter out file paths / junk
            if text_str.endswith('.mp4') or text_str.endswith('.wav') or text_str.endswith('.pt'):
                continue
            
            # Check if likely a path (contains slashes but no spaces/Vietnamese)
            if ('/' in text_str or '\\' in text_str) and ' ' not in text_str:
                vi_chars = 'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'
                if not any(c in text_str.lower() for c in vi_chars):
                    continue 
            
            return text_str

    return ""
