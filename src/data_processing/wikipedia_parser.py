"""Wikipedia XML dump parser for extracting clean article text."""

import bz2
import xml.etree.ElementTree as ET
import re
import json
from pathlib import Path
from typing import Generator, Dict, List
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class WikipediaParser:
    """Parser for Wikipedia XML dumps."""
    
    def __init__(self, config: dict):
        self.config = config
        self.min_length = config['processing']['min_article_length']
        self.max_length = config['processing']['max_article_length']
        
        # Regex patterns for cleaning
        self.cleanup_patterns = [
            (r'\{\{[^}]*\}\}', ''),  # Remove templates
            (r'\[\[Category:[^\]]*\]\]', ''),  # Remove categories
            (r'\[\[File:[^\]]*\]\]', ''),  # Remove file links
            (r'\[\[Image:[^\]]*\]\]', ''),  # Remove image links
            (r'<ref[^>]*>.*?</ref>', ''),  # Remove references
            (r'<ref[^>]*/?>', ''),  # Remove self-closing refs
            (r'&lt;.*?&gt;', ''),  # Remove HTML entities
            (r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2'),  # Clean internal links
            (r'\[http[s]?://[^\s]+ ([^\]]*)\]', r'\1'),  # Clean external links
            (r"'''([^']*?)'''", r'\1'),  # Remove bold markup
            (r"''([^']*?)''", r'\1'),  # Remove italic markup
            (r'\n+', ' '),  # Replace multiple newlines with space
            (r'\s+', ' '),  # Replace multiple spaces with single space
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean Wikipedia markup from text."""
        if not text:
            return ""
        
        # Apply all cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text, flags=re.DOTALL | re.IGNORECASE)
        
        return text.strip()
    
    def is_valid_article(self, title: str, text: str) -> bool:
        """Check if article meets quality criteria."""
        if not title or not text:
            return False
        
        # Skip redirects
        if text.lower().startswith('#redirect'):
            return False
        
        # Skip disambiguation pages
        if '(disambiguation)' in title.lower():
            return False
        
        # Skip if too short or too long
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Skip special pages
        skip_prefixes = ['wikipedia:', 'template:', 'category:', 'file:', 'help:', 'user:']
        if any(title.lower().startswith(prefix) for prefix in skip_prefixes):
            return False
        
        return True
    
    def parse_xml_stream(self, file_path: str) -> Generator[Dict[str, str], None, None]:
        """Parse Wikipedia XML dump and yield clean articles."""
        logger.info(f"Starting to parse {file_path}")
        
        # Open compressed file
        with bz2.open(file_path, 'rt', encoding='utf-8') as file:
            current_article = {}
            inside_page = False
            inside_text = False
            text_content = []
            
            for line_num, line in enumerate(tqdm(file, desc="Parsing XML", unit="lines")):
                line = line.strip()
                
                if '<page>' in line:
                    inside_page = True
                    current_article = {}
                    continue
                
                if '</page>' in line and inside_page:
                    # Process completed article
                    if 'title' in current_article and 'text' in current_article:
                        title = current_article['title']
                        text = self.clean_text(current_article['text'])
                        
                        if self.is_valid_article(title, text):
                            yield {
                                'title': title,
                                'text': text,
                                'id': current_article.get('id', ''),
                                'length': len(text)
                            }
                    
                    inside_page = False
                    continue
                
                if inside_page:
                    # Extract title
                    if '<title>' in line and '</title>' in line:
                        title_match = re.search(r'<title>(.*?)</title>', line)
                        if title_match:
                            current_article['title'] = title_match.group(1)
                    
                    # Extract ID
                    elif '<id>' in line and '</id>' in line and 'id' not in current_article:
                        id_match = re.search(r'<id>(\d+)</id>', line)
                        if id_match:
                            current_article['id'] = id_match.group(1)
                    
                    # Handle text content
                    elif '<text' in line:
                        inside_text = True
                        text_content = []
                        # Extract any text on the same line
                        text_start = line.find('>')
                        if text_start != -1:
                            text_content.append(line[text_start + 1:])
                    
                    elif '</text>' in line and inside_text:
                        # Add final text and close
                        end_pos = line.find('</text>')
                        if end_pos != -1:
                            text_content.append(line[:end_pos])
                        current_article['text'] = '\n'.join(text_content)
                        inside_text = False
                    
                    elif inside_text:
                        text_content.append(line)
                
                # Progress update every 100k lines
                if line_num % 100000 == 0 and line_num > 0:
                    logger.info(f"Processed {line_num:,} lines")
    
    def save_articles(self, articles: List[Dict[str, str]], output_file: str):
        """Save articles to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} articles to {output_file}")
    
    def process_dump(self, input_file: str, output_file: str, max_articles: int = None) -> Dict[str, int]:
        """Process Wikipedia dump and save clean articles."""
        articles = []
        stats = {
            'total_processed': 0,
            'valid_articles': 0,
            'skipped': 0
        }
        
        for article in self.parse_xml_stream(input_file):
            stats['total_processed'] += 1
            
            if max_articles and len(articles) >= max_articles:
                break
                
            articles.append(article)
            stats['valid_articles'] += 1
            
            # Save in batches to avoid memory issues
            if len(articles) >= self.config['processing']['batch_size']:
                batch_file = output_file.replace('.json', f'_batch_{stats["valid_articles"]//self.config["processing"]["batch_size"]}.json')
                self.save_articles(articles, batch_file)
                articles = []
        
        # Save remaining articles
        if articles:
            final_file = output_file.replace('.json', f'_final.json')
            self.save_articles(articles, final_file)
        
        stats['skipped'] = stats['total_processed'] - stats['valid_articles']
        
        logger.info(f"Processing complete: {stats}")
        return stats


def main():
    """Test the parser with a small sample."""
    import sys
    sys.path.append('..')
    from src.utils import load_config, setup_logging
    
    config = load_config()
    setup_logging(config)
    
    parser = WikipediaParser(config)
    
    # Test with first 1000 articles
    input_file = config['data']['raw_dump']
    output_file = f"{config['data']['processed_dir']}sample_articles.json"
    
    stats = parser.process_dump(input_file, output_file, max_articles=1000)
    print(f"Parser test complete: {stats}")


if __name__ == "__main__":
    main()