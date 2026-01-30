#simple UI on streamlit

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime


sys.path.append(str(Path(__file__).parent))

from nlp.intent_classifier import IntentClassifier
from nlp.entity_extractor import EntityExtractor
from nlp.action_decider import ActionDecider


st.set_page_config(
    page_title="Logistics AI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding-top: 2rem;
        max-width: 900px;
    }
    
    /* Header */
    .header-container {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        opacity: 0.7;
        font-weight: 400;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    /* Results container */
    .results-box {
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Intent section */
    .intent-section {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
        margin-bottom: 2rem;
    }
    .intent-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.6;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .intent-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 1rem 0;
        letter-spacing: -0.5px;
    }
    
    /* Confidence bar */
    .confidence-container {
        max-width: 400px;
        margin: 1.5rem auto 0;
    }
    
    .confidence-text {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-bottom: 0.7rem;
        text-align: left;
    }
    
    .confidence-bar-bg {
        height: 10px;
        background: rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    
    .confidence-bar-fill {
        height: 100%;
        background: #2563eb;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .confidence-percent {
        font-size: 1.1rem;
        font-weight: 700;
        text-align: right;
    }
    
    /* Section headers */
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.6;
        margin: 2rem 0 1.2rem 0;
    }
    
    /* Entity items */
    .entity-list {
        display: grid;
        gap: 0.7rem;
    }
    
    .entity-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    .entity-name {
        font-size: 0.95rem;
        opacity: 0.8;
        font-weight: 500;
    }
    
    .entity-val {
        font-size: 0.95rem;
        font-weight: 600;
    }
    
    .found {
        color: #10b981;
    }
    
    .missing {
        opacity: 0.4;
        font-style: italic;
        font-weight: 400;
    }
    
    /* Action box */
    .action-box {
        margin-top: 2rem;
        padding: 1.3rem;
        border-radius: 8px;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    .action-header {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.8;
        margin-bottom: 0.7rem;
    }
    
    .action-text {
        font-size: 1rem;
        line-height: 1.6;
        opacity: 0.9;
    }
    
    /* Info box */
    .info-notice {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    .info-notice p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.6;
        opacity: 0.9;
    }
    
    /* Examples */
    .examples-header {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    
    .example-box {
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.7rem;
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
        font-size: 0.95rem;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .example-box:hover {
        background: rgba(128, 128, 128, 0.1);
        border-color: rgba(128, 128, 128, 0.2);
        transform: translateX(4px);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Animation */
    .fade-in {
        animation: fadeIn 0.4s ease-out;
    }
    
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(15px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load ML models"""
    try:
        classifier = IntentClassifier()
        model_path = Path(__file__).parent / 'models' / 'intent_model.pkl'
        
        if model_path.exists():
            classifier.load(str(model_path))
            extractor = EntityExtractor()
            decider = ActionDecider()
            return classifier, extractor, decider, None
        else:
            return None, None, None, "Model not found. Please train the model first."
    except Exception as e:
        return None, None, None, f"Error loading models: {str(e)}"


def display_results(intent_result, entities, action):
    """Display results"""
    
    st.markdown('<div class="results-box fade-in">', unsafe_allow_html=True)
    
    # Intent Section
    confidence = intent_result['confidence']
    st.markdown(f'''
        <div class="intent-section">
            <div class="intent-label">Detected Intent</div>
            <div class="intent-value">{intent_result['intent']}</div>
            <div class="confidence-container">
                <div class="confidence-text">Confidence Score</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width: {confidence*100}%"></div>
                </div>
                <div class="confidence-percent">{confidence:.0%}</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Entities Section
    st.markdown('<div class="section-title">Extracted Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="entity-list">', unsafe_allow_html=True)
    
    entity_labels = {
        'pickup_location': '📍 Pickup Location',
        'drop_location': '📍 Drop Location',
        'weight_kg': '⚖️ Weight',
        'packages': '📦 Packages',
        'pickup_time': '⏰ Pickup Time',
        'fragile': '⚠️ Fragile',
        'payment_mode': '💳 Payment Mode',
        'phone_number': '📞 Phone Number'
    }
    
    for key, label in entity_labels.items():
        value = entities.get(key)
        
        if value is not None and value is not False:
            if isinstance(value, bool):
                display_val = "Yes"
            else:
                display_val = f"{value} kg" if key == "weight_kg" else str(value)
            css_class = "found"
            display_text = f'✓ {display_val}'
        else:
            css_class = "missing"
            display_text = "Not provided"
        
        st.markdown(
            f'<div class="entity-item">'
            f'<span class="entity-name">{label}</span>'
            f'<span class="entity-val {css_class}">{display_text}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action Section
    if 'message' in action:
        st.markdown(
            f'<div class="action-box">'
            f'<div class="action-header">Next Step</div>'
            f'<div class="action-text">{action["message"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

   
    st.markdown("---")
    with st.expander("🔍 View JSON Output", expanded=False):
        import json as json_module
        json_output = {
            "query": st.session_state.get('last_query', ''),
            "intent": {
                "intent": intent_result['intent'],
                "confidence": intent_result['confidence']
            },
            "entities": entities,
            "next_action": action
        }
        st.json(json_output)
        

        json_str = json_module.dumps(json_output, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="query_result.json",
            mime="application/json",
            use_container_width=True
        )





def main():
    # Header
    st.markdown('''
        <div class="header-container">
            <h1 class="main-title"> Logistics AI Assistant</h1>
        </div>
    ''', unsafe_allow_html=True)
    
    # Load models
    classifier, extractor, decider, error = load_models()
    
    if error:
        st.error(f"{error}")
        st.info("**Setup:** Navigate to the `nlp` folder and run `python intent_classifier.py` to train the model.")
        return
    
    
    # Query input
    query = st.text_area(
        "Your Query",
        placeholder="Example: Bhai pickup karna hai Andheri se Powai, 2 boxes hai",
        height=100,
        label_visibility="collapsed"
    )
    
    # Process button
    process_button = st.button("Process Query", type="primary")
    
    # Process query
    if process_button:
        if query.strip():
            with st.spinner("Processing your query..."):
                intent_result = classifier.predict(query)
                entities = extractor.extract(query)
                action = decider.decide_action(
                    intent=intent_result['intent'],
                    entities=entities,
                    confidence=intent_result['confidence']
                )
                
                display_results(intent_result, entities, action)
        else:
            st.warning(" Please enter a query")
    
    # Examples section
    if not process_button or not query.strip():
        st.markdown("---")
        st.markdown('<div class="examples-header">Try these examples:</div>', unsafe_allow_html=True)
        
        examples = [
            "Bhai price batao Mumbai to Pune 10kg",
            "Pickup karna hai Andheri se Powai, 2 boxes hai",
            "Mera order track karo urgent",
            "COD available hai kya",
            "Delivery late hai complaint register karo"
        ]
        
        for example in examples:
            st.markdown(f'<div class="example-box">{example}</div>', unsafe_allow_html=True)
        
        st.caption(" Copy any example and paste into the query box above")


if __name__ == "__main__":
    main()