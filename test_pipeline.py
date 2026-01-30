"""
Complete Pipeline Test Script
Demonstrates end-to-end functionality of the Voice-to-Action system
"""


from nlp.intent_classifier import IntentClassifier
from nlp.entity_extractor import EntityExtractor
from nlp.action_decider import ActionDecider
import json


def print_separator(title=""):
    """Print a nice separator"""
    print("\n" + "="*70)
    if title:
        print(f"{title.center(70)}")
        print("="*70)


def print_result(query, intent_result, entities, action):
    """Pretty print the pipeline results"""
    print(f"\n📝 Query: {query}")
    print(f"\n🎯 Intent: {intent_result['intent']} (confidence: {intent_result['confidence']})")
    
    print(f"\n📦 Entities:")
    for key, value in entities.items():
        print(f"   {key}: {value}")
    
    print(f"\n⚡ Next Action: {action['next_action']}")
    if 'message' in action:
        print(f"   Message: {action['message']}")
    if 'missing_fields' in action:
        print(f"   Missing: {action['missing_fields']}")


def main():
    print_separator("VOICE-TO-ACTION AI PIPELINE TEST")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    intent_classifier = IntentClassifier()
    entity_extractor = EntityExtractor()
    action_decider = ActionDecider()
    
    # Load trained model
    model_path = 'models/intent_model.pkl'
    try:
        intent_classifier.load(model_path)
        print("✅ Intent classifier loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please train the model first by running: python nlp/intent_classifier.py")
        return
    
    # Test queries
    test_queries = [
        "Bhai price batao Mumbai to Pune 10kg",
        "Pickup karna hai Andheri se Powai, 2 boxes hai",
        "Kal morning pickup possible hai kya 9876543210",
        "Mera order track karo",
        "COD available hai kya",
        "Parcel fragile hai handle carefully Mumbai to Delhi",
        "Delivery late hai complaint register karo",
        "Customer care se baat karni hai urgent",
        "Invoice upload karna hai",
        "Noida se Ghaziabad serviceable hai kya"
    ]
    
    print_separator("RUNNING TEST QUERIES")
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print_separator(f"Test Case {i}")
        
        # Step 1: Intent Classification
        intent_result = intent_classifier.predict(query)
        
        # Step 2: Entity Extraction
        entities = entity_extractor.extract(query)
        
        # Step 3: Action Decision
        action = action_decider.decide_action(
            intent=intent_result['intent'],
            entities=entities,
            confidence=intent_result['confidence']
        )
        
        # Print results
        print_result(query, intent_result, entities, action)
        
        # Store for summary
        results.append({
            'query': query,
            'intent': intent_result,
            'entities': entities,
            'action': action
        })
    
    # Summary statistics
    print_separator("SUMMARY STATISTICS")
    
    intents_found = {}
    total_entities = 0
    non_null_entities = 0
    
    for result in results:
        intent = result['intent']['intent']
        intents_found[intent] = intents_found.get(intent, 0) + 1
        
        for key, value in result['entities'].items():
            total_entities += 1
            if value is not None and value != False:
                non_null_entities += 1
    
    print(f"\n📊 Total Queries: {len(results)}")
    print(f"\n🎯 Intent Distribution:")
    for intent, count in sorted(intents_found.items()):
        print(f"   {intent}: {count}")
    
    entity_extraction_rate = (non_null_entities / total_entities) * 100
    print(f"\n📦 Entity Extraction Rate: {entity_extraction_rate:.1f}%")
    print(f"   Non-null entities: {non_null_entities}/{total_entities}")
    
    avg_confidence = sum(r['intent']['confidence'] for r in results) / len(results)
    print(f"\n📈 Average Confidence: {avg_confidence:.2f}")
    
    # Action type distribution
    action_types = {}
    for result in results:
        action = result['action']['next_action']
        action_types[action] = action_types.get(action, 0) + 1
    
    print(f"\n⚡ Action Distribution:")
    for action, count in sorted(action_types.items()):
        print(f"   {action}: {count}")
    
    print_separator("TEST COMPLETE")
    print("\n✅ All tests completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Review the results above")
    print("   2. Check EVALUATION_REPORT.md for detailed analysis")
    print("   3. Start the API: cd api && uvicorn main:app --reload")
    print("   4. Test API at: http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()