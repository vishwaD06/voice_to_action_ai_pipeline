"""
Entity Extraction Module
Uses spaCy NER + regex/keyword rules for structured information extraction
"""

import re
import spacy
from typing import Dict, Optional, List


class EntityExtractor:
    """
    Rule-based entity extraction for logistics queries
    """
    
    def __init__(self):
        """
        Initialize spaCy model and patterns
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Location keywords (common Indian cities and areas)
        self.location_keywords = [
            'mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata',
            'hyderabad', 'ahmedabad', 'gurgaon', 'noida', 'thane', 'navi mumbai',
            'andheri', 'powai', 'bandra', 'worli', 'kurla', 'ghatkopar',
            'khargar', 'vashi', 'panvel', 'whitefield', 'koramangala',
            'indiranagar', 'malleswaram', 'mg road', 'connaught place',
            'saket', 'vasant kunj', 'dwarka', 'rohini', 'ghaziabad',
            'faridabad', 'kashmir', 'tier'
        ]
        
        # Time keywords
        self.time_keywords = {
            'morning': 'morning',
            'afternoon': 'afternoon',
            'evening': 'evening',
            'night': 'night',
            'kal': 'tomorrow',
            'aaj': 'today',
            'parso': 'day_after_tomorrow'
        }
        
        # Payment mode keywords
        self.payment_keywords = {
            'cod': 'COD',
            'cash on delivery': 'COD',
            'cash': 'COD',
            'prepaid': 'prepaid',
            'online': 'prepaid',
            'upi': 'prepaid',
            'card': 'prepaid'
        }
    
    def extract_locations(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract pickup and drop locations using spaCy NER + keyword matching
        """
        text_lower = text.lower()
        locations = []
        
        # Use spaCy NER for GPE (Geo-Political Entity)
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                locations.append(ent.text)
        
        # Keyword-based extraction for Indian locations
        for keyword in self.location_keywords:
            if keyword in text_lower:
                locations.append(keyword.title())
        
        # Remove duplicates while preserving order
        locations = list(dict.fromkeys(locations))
        
        # Try to identify pickup vs drop using context
        pickup_location = None
        drop_location = None
        
        if len(locations) >= 2:
            # Look for "se" (from) and "to" patterns
            if ' se ' in text_lower or ' from ' in text_lower:
                pickup_location = locations[0]
                drop_location = locations[1]
            elif ' to ' in text_lower:
                # First location is pickup if "to" is present
                pickup_location = locations[0]
                drop_location = locations[1]
            else:
                # Default: first is pickup, second is drop
                pickup_location = locations[0]
                drop_location = locations[1]
        elif len(locations) == 1:
            # Only one location found
            if 'pickup' in text_lower or 'se' in text_lower:
                pickup_location = locations[0]
            elif 'drop' in text_lower or 'delivery' in text_lower:
                drop_location = locations[0]
        
        return {
            'pickup_location': pickup_location,
            'drop_location': drop_location
        }
    
    def extract_weight(self, text: str) -> Optional[float]:
        """
        Extract weight in kg using regex
        """
        text_lower = text.lower()
        
        # Pattern: number followed by kg/kgs/kilogram
        patterns = [
            r'(\d+\.?\d*)\s*kg',
            r'(\d+\.?\d*)\s*kgs',
            r'(\d+\.?\d*)\s*kilogram',
            r'(\d+\.?\d*)\s*kilos'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))
        
        return None
    
    def extract_packages(self, text: str) -> Optional[int]:
        """
        Extract number of packages/boxes
        """
        text_lower = text.lower()
        
        # Pattern: number followed by box/boxes/packages/parcels
        patterns = [
            r'(\d+)\s*box',
            r'(\d+)\s*package',
            r'(\d+)\s*parcel',
            r'(\d+)\s*item'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    def extract_time(self, text: str) -> Optional[str]:
        """
        Extract pickup time information
        """
        text_lower = text.lower()
        
        # Check for time keywords
        for keyword, value in self.time_keywords.items():
            if keyword in text_lower:
                return value
        
        # Check for specific times (e.g., "6 PM", "4:30")
        time_patterns = [
            r'(\d{1,2})\s*(am|pm)',
            r'(\d{1,2}):(\d{2})\s*(am|pm)?',
            r'(\d{1,2})\s*baje'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(0)
        
        return None
    
    def extract_fragile(self, text: str) -> bool:
        """
        Check if package is fragile
        """
        text_lower = text.lower()
        fragile_keywords = ['fragile', 'breakable', 'handle carefully', 'delicate']
        
        return any(keyword in text_lower for keyword in fragile_keywords)
    
    def extract_payment_mode(self, text: str) -> Optional[str]:
        """
        Extract payment mode (COD or prepaid)
        """
        text_lower = text.lower()
        
        for keyword, mode in self.payment_keywords.items():
            if keyword in text_lower:
                return mode
        
        return None
    
    def extract_phone_number(self, text: str) -> Optional[str]:
        """
        Extract Indian phone number (10 digits)
        """
        # Pattern for Indian phone numbers
        patterns = [
            r'\b[6-9]\d{9}\b',  # 10 digit starting with 6-9
            r'\+91[\s-]?[6-9]\d{9}\b'  # With +91
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None
    
    def extract(self, text: str) -> Dict:
        """
        Main extraction method - extracts all entities
        
        Args:
            text: User query
            
        Returns:
            dict: All extracted entities
        """
        locations = self.extract_locations(text)
        
        entities = {
            'pickup_location': locations['pickup_location'],
            'drop_location': locations['drop_location'],
            'weight_kg': self.extract_weight(text),
            'packages': self.extract_packages(text),
            'pickup_time': self.extract_time(text),
            'fragile': self.extract_fragile(text),
            'payment_mode': self.extract_payment_mode(text),
            'phone_number': self.extract_phone_number(text)
        }
        
        return entities


# Testing script
if __name__ == "__main__":
    print("="*60)
    print("ENTITY EXTRACTION TESTING")
    print("="*60)
    
    extractor = EntityExtractor()
    
    test_queries = [
        "Bhai pickup karna hai Andheri se Powai, 2 boxes hai",
        "Rate batao Mumbai to Pune 10kg fragile package",
        "Kal morning pickup possible hai kya 9876543210",
        "COD me 3 parcels bhejne hai Delhi se Bangalore",
        "Evening 6 baje pickup kar lo Gurgaon se"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        entities = extractor.extract(query)
        print(f"Entities:")
        for key, value in entities.items():
            print(f"  {key}: {value}")