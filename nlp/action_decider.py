"""
Action Decider Module
Rule-based logic to determine next system action based on intent + entities
"""

from typing import Dict, List, Optional


class ActionDecider:
    """
    Decision engine to determine what the system should do next
    Based on intent classification and extracted entities
    """
    
    # Define required fields for each intent
    INTENT_REQUIREMENTS = {
        'CHECK_RATE': {
            'required': ['pickup_location', 'drop_location', 'weight_kg'],
            'optional': ['packages']
        },
        'CHECK_SERVICEABILITY': {
            'required': ['drop_location'],
            'optional': ['pickup_location']
        },
        'BOOK_PICKUP': {
            'required': ['pickup_location', 'drop_location', 'packages'],
            'optional': ['pickup_time', 'weight_kg', 'phone_number', 'payment_mode']
        },
        'TRACK_ORDER': {
            'required': [],  # AWB number would be extracted differently
            'optional': ['phone_number']
        },
        'CANCEL_ORDER': {
            'required': [],  # Order ID would be from context/session
            'optional': []
        },
        'RESCHEDULE_PICKUP': {
            'required': ['pickup_time'],
            'optional': ['pickup_location']
        },
        'RAISE_COMPLAINT': {
            'required': [],
            'optional': ['phone_number']
        },
        'CONNECT_TO_AGENT': {
            'required': [],
            'optional': []
        },
        'PAYMENT_QUERY': {
            'required': [],
            'optional': []
        },
        'DOCUMENT_UPLOAD_QUERY': {
            'required': [],
            'optional': []
        }
    }
    
    def __init__(self):
        pass
    
    def find_missing_fields(self, intent: str, entities: Dict) -> List[str]:
        """
        Identify which required fields are missing for given intent
        
        Args:
            intent: Classified intent
            entities: Extracted entities dict
            
        Returns:
            List of missing field names
        """
        if intent not in self.INTENT_REQUIREMENTS:
            return []
        
        requirements = self.INTENT_REQUIREMENTS[intent]
        required_fields = requirements.get('required', [])
        
        missing = []
        for field in required_fields:
            if entities.get(field) is None:
                missing.append(field)
        
        return missing
    
    def decide_action(self, intent: str, entities: Dict, confidence: float = 1.0) -> Dict:
        """
        Decide next system action based on intent and entities
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            confidence: Intent classification confidence
            
        Returns:
            dict: Next action information
        """
    
        
        # Find missing fields
        missing_fields = self.find_missing_fields(intent, entities)
        
        # Decision logic based on intent
        if intent == 'CHECK_RATE':
            if missing_fields:
                return {
                    'next_action': 'ASK_MISSING_FIELDS',
                    'missing_fields': missing_fields,
                    'message': self._generate_missing_fields_message(missing_fields)
                }
            else:
                return {
                    'next_action': 'CALCULATE_RATE',
                    'message': 'Fetching rate information...',
                    'api_call': 'pricing_api',
                    'parameters': {
                        'from': entities['pickup_location'],
                        'to': entities['drop_location'],
                        'weight': entities['weight_kg']
                    }
                }
        
        elif intent == 'CHECK_SERVICEABILITY':
            if missing_fields:
                return {
                    'next_action': 'ASK_MISSING_FIELDS',
                    'missing_fields': missing_fields,
                    'message': self._generate_missing_fields_message(missing_fields)
                }
            else:
                return {
                    'next_action': 'CHECK_SERVICE_AREA',
                    'message': 'Checking serviceability...',
                    'api_call': 'serviceability_api',
                    'parameters': {
                        'location': entities['drop_location']
                    }
                }
        
        elif intent == 'BOOK_PICKUP':
            if missing_fields:
                return {
                    'next_action': 'ASK_MISSING_FIELDS',
                    'missing_fields': missing_fields,
                    'message': self._generate_missing_fields_message(missing_fields)
                }
            else:
                # Check for recommended optional fields
                recommended = []
                if not entities.get('pickup_time'):
                    recommended.append('pickup_time')
                if not entities.get('phone_number'):
                    recommended.append('phone_number')
                
                if recommended:
                    return {
                        'next_action': 'ASK_OPTIONAL_FIELDS',
                        'optional_fields': recommended,
                        'message': f"I can book your pickup. Would you like to specify {', '.join(recommended)}?",
                        'can_proceed': True
                    }
                else:
                    return {
                        'next_action': 'CREATE_BOOKING',
                        'message': 'Creating your pickup booking...',
                        'api_call': 'booking_api',
                        'parameters': entities
                    }
        
        elif intent == 'TRACK_ORDER':
            return {
                'next_action': 'ASK_TRACKING_INFO',
                'message': 'Please provide your AWB number or order ID to track',
                'required_info': 'awb_number'
            }
        
        elif intent == 'CANCEL_ORDER':
            return {
                'next_action': 'ASK_ORDER_ID',
                'message': 'Please provide your order ID or AWB number to cancel',
                'required_info': 'order_id'
            }
        
        elif intent == 'RESCHEDULE_PICKUP':
            if missing_fields:
                return {
                    'next_action': 'ASK_MISSING_FIELDS',
                    'missing_fields': missing_fields,
                    'message': 'When would you like to reschedule the pickup?'
                }
            else:
                return {
                    'next_action': 'ASK_ORDER_ID',
                    'message': 'Please provide your booking ID to reschedule',
                    'new_time': entities.get('pickup_time')
                }
        
        elif intent == 'RAISE_COMPLAINT':
            return {
                'next_action': 'CREATE_TICKET',
                'message': 'I will create a complaint ticket. Please describe your issue.',
                'ticket_type': 'complaint',
                'contact': entities.get('phone_number')
            }
        
        elif intent == 'CONNECT_TO_AGENT':
            return {
                'next_action': 'TRANSFER_TO_AGENT',
                'message': 'Connecting you to a customer service agent...',
                'priority': 'normal'
            }
        
        elif intent == 'PAYMENT_QUERY':
            return {
                'next_action': 'PROVIDE_PAYMENT_INFO',
                'message': 'We accept COD, UPI, cards, and online payment. Which option would you prefer?',
                'available_modes': ['COD', 'UPI', 'Card', 'Net Banking']
            }
        
        elif intent == 'DOCUMENT_UPLOAD_QUERY':
            return {
                'next_action': 'PROVIDE_UPLOAD_LINK',
                'message': 'You can upload documents through our portal or app. What document do you need to upload?',
                'upload_options': ['Invoice', 'KYC', 'GST Certificate', 'ID Proof']
            }
        
        else:
            return {
                'next_action': 'UNKNOWN',
                'message': 'I am not sure how to help with that. Please contact customer support.',
                'intent': intent
            }
    
    def _generate_missing_fields_message(self, missing_fields: List[str]) -> str:
        """
        Generate user-friendly message for missing fields
        """
        field_messages = {
            'pickup_location': 'pickup location',
            'drop_location': 'delivery location',
            'weight_kg': 'package weight (in kg)',
            'packages': 'number of packages',
            'pickup_time': 'preferred pickup time',
            'phone_number': 'contact number'
        }
        
        readable_fields = [field_messages.get(f, f) for f in missing_fields]
        
        if len(readable_fields) == 1:
            return f"Please provide {readable_fields[0]}."
        else:
            return f"Please provide {', '.join(readable_fields[:-1])} and {readable_fields[-1]}."


# Testing script
if __name__ == "__main__":
    print("="*60)
    print("ACTION DECIDER TESTING")
    print("="*60)
    
    decider = ActionDecider()
    
    # Test cases
    test_cases = [
        {
            'intent': 'CHECK_RATE',
            'entities': {
                'pickup_location': 'Mumbai',
                'drop_location': None,
                'weight_kg': 10,
                'packages': None,
                'pickup_time': None,
                'fragile': False,
                'payment_mode': None,
                'phone_number': None
            },
            'confidence': 0.95
        },
        {
            'intent': 'BOOK_PICKUP',
            'entities': {
                'pickup_location': 'Andheri',
                'drop_location': 'Powai',
                'weight_kg': None,
                'packages': 2,
                'pickup_time': None,
                'fragile': False,
                'payment_mode': None,
                'phone_number': None
            },
            'confidence': 0.87
        },
        {
            'intent': 'TRACK_ORDER',
            'entities': {
                'pickup_location': None,
                'drop_location': None,
                'weight_kg': None,
                'packages': None,
                'pickup_time': None,
                'fragile': False,
                'payment_mode': None,
                'phone_number': None
            },
            'confidence': 0.92
        },
        {
            'intent': 'RAISE_COMPLAINT',
            'entities': {
                'pickup_location': None,
                'drop_location': None,
                'weight_kg': None,
                'packages': None,
                'pickup_time': None,
                'fragile': False,
                'payment_mode': None,
                'phone_number': '9876543210'
            },
            'confidence': 0.88
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}")
        print(f"{'='*60}")
        print(f"Intent: {test['intent']} (confidence: {test['confidence']})")
        print(f"Entities: {test['entities']}")
        
        action = decider.decide_action(
            test['intent'],
            test['entities'],
            test['confidence']
        )
        
        print(f"\nNext Action:")
        for key, value in action.items():
            print(f"  {key}: {value}")