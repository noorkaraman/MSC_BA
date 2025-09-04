import streamlit as st
import pandas as pd
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="MND Device Onboarding",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for MND-friendly onboarding interface
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .question-container {
        background-color: #f8f9ff;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #4f8ef7;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .question-title {
        font-size: 18px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
        line-height: 1.4;
    }
    
    .persona-selector {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    
    .stTextArea textarea {
        font-size: 16px !important;
        line-height: 1.5 !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
        min-height: 80px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4f8ef7 !important;
        box-shadow: 0 0 0 3px rgba(79, 142, 247, 0.1) !important;
    }
    
    .stSelectbox select {
        font-size: 16px !important;
        padding: 12px !important;
        border-radius: 8px !important;
    }
    
    .progress-indicator {
        background-color: #e9ecef;
        height: 8px;
        border-radius: 4px;
        margin: 20px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        background-color: #4f8ef7;
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    .save-button {
        background-color: #28a745;
        color: white;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        width: 100%;
        margin: 20px 0;
    }
    
    .save-button:hover {
        background-color: #218838;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        text-align: center;
    }
    
    .navigation-buttons {
        display: flex;
        gap: 15px;
        margin: 20px 0;
    }
    
    .nav-button {
        flex: 1;
        padding: 12px 20px;
        font-size: 16px;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        background-color: #f8f9fa;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .nav-button:hover {
        background-color: #e9ecef;
        border-color: #4f8ef7;
    }
    
    .nav-button.primary {
        background-color: #4f8ef7;
        color: white;
        border-color: #4f8ef7;
    }
    
    .nav-button.primary:hover {
        background-color: #3d7bd7;
    }
</style>
""", unsafe_allow_html=True)

# Load onboarding data
@st.cache_data
def load_onboarding_data():
    """Load the onboarding questions and existing persona responses"""
    try:
        onboarding_df = pd.read_csv("data/Onboarding_QnA.csv")
        return onboarding_df
    except FileNotFoundError:
        try:
            onboarding_df = pd.read_csv("Onboarding_QnA.csv")
            return onboarding_df
        except FileNotFoundError:
            st.error("Could not find Onboarding_QnA.csv file. Please ensure it's in the correct location.")
            return None

def get_persona_responses():
    """Get the existing persona responses as examples"""
    df = load_onboarding_data()
    if df is None:
        return {}
    
    personas = {}
    for _, row in df.iterrows():
        persona_name = row.get('Persona', '')
        # Convert to string and check if it's valid
        if pd.notna(persona_name) and str(persona_name).strip():
            personas[str(persona_name)] = dict(row)
    
    return personas

def get_question_columns():
    """Get all question columns from the onboarding data (excluding Persona column)"""
    df = load_onboarding_data()
    if df is None:
        return []
    
    # Remove 'Persona' column and any empty columns
    columns = [col for col in df.columns if col != 'Persona' and not df[col].isna().all()]
    return columns

# Initialize session state
def initialize_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'persona_name' not in st.session_state:
        st.session_state.persona_name = ""
    if 'selected_example_persona' not in st.session_state:
        st.session_state.selected_example_persona = "Sarah Ahmed"

def save_responses():
    """Save the current responses to a CSV file"""
    if not st.session_state.persona_name.strip():
        st.error("Please enter a persona name before saving.")
        return False
    
    # Create a new row with the responses
    new_row = {'Persona': st.session_state.persona_name}
    new_row.update(st.session_state.responses)
    
    # Convert to DataFrame
    df = pd.DataFrame([new_row])
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"onboarding_responses_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    st.success(f"Responses saved successfully to {filename}")
    return True

def main():
    st.title("📋 MND Device Onboarding")
    st.markdown("*Complete the setup questionnaire for your communication device*")
    
    # Initialize session state
    initialize_session_state()
    
    # Load data
    personas = get_persona_responses()
    questions = get_question_columns()
    
    if not questions:
        st.error("No questions found. Please check the onboarding data file.")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Persona name input
        st.session_state.persona_name = st.text_input(
            "Patient/User Name:",
            value=st.session_state.persona_name,
            placeholder="Enter patient name",
            help="This will be the name for this configuration"
        )
        
        # Example persona selector
        if personas:
            st.session_state.selected_example_persona = st.selectbox(
                "Use responses from example persona:",
                options=list(personas.keys()),
                index=list(personas.keys()).index(st.session_state.selected_example_persona) 
                if st.session_state.selected_example_persona in personas else 0,
                help="Pre-fill answers with examples from existing personas"
            )
            
            if st.button("Load Example Responses"):
                selected_persona_data = personas[st.session_state.selected_example_persona]
                for question in questions:
                    if question in selected_persona_data:
                        st.session_state.responses[question] = str(selected_persona_data[question])
                st.success(f"Loaded responses from {st.session_state.selected_example_persona}")
                st.rerun()
        
        # Progress indicator
        st.markdown("### Progress")
        progress = len([q for q in questions if q in st.session_state.responses and st.session_state.responses[q].strip()]) / len(questions)
        st.progress(progress)
        st.write(f"{int(progress * 100)}% Complete")
        
        # Navigation
        st.markdown("### Navigation")
        questions_per_page = 5
        total_pages = (len(questions) + questions_per_page - 1) // questions_per_page
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous", disabled=st.session_state.current_page == 0):
                st.session_state.current_page = max(0, st.session_state.current_page - 1)
                st.rerun()
        
        with col2:
            if st.button("Next →", disabled=st.session_state.current_page >= total_pages - 1):
                st.session_state.current_page = min(total_pages - 1, st.session_state.current_page + 1)
                st.rerun()
        
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
        
        # Save button
        st.markdown("---")
        if st.button("💾 Save All Responses", type="primary"):
            save_responses()
    
    # Main content area
    st.markdown(f"""
    <div class="section-header">
        <h2>Onboarding Questions - Page {st.session_state.current_page + 1}</h2>
        <p>Please answer the following questions to personalize your communication device</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate questions for current page
    questions_per_page = 5
    start_idx = st.session_state.current_page * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(questions))
    current_questions = questions[start_idx:end_idx]
    
    # Display questions for current page
    for i, question in enumerate(current_questions):
        st.markdown(f"""
        <div class="question-container">
            <div class="question-title">
                {start_idx + i + 1}. {question}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get pre-filled value
        current_value = st.session_state.responses.get(question, "")
        
        # If no current value and example persona selected, use example
        if not current_value and personas and st.session_state.selected_example_persona in personas:
            example_value = personas[st.session_state.selected_example_persona].get(question, "")
            # Convert to string and check if it's valid
            if pd.notna(example_value):
                example_str = str(example_value).strip()
                if example_str and example_str.lower() != 'nan':
                    current_value = example_str
        
        # Text area for response
        response = st.text_area(
            f"Your response:",
            value=current_value,
            key=f"question_{question}",
            height=100,
            placeholder="Enter your response here...",
            help=f"Answer for: {question}"
        )
        
        # Update session state
        st.session_state.responses[question] = response
        
        # Show example if different from current
        if (personas and st.session_state.selected_example_persona in personas and 
            question in personas[st.session_state.selected_example_persona]):
            example_response = personas[st.session_state.selected_example_persona][question]
            if pd.notna(example_response):
                example_str = str(example_response).strip()
                if (example_str and example_str.lower() != 'nan' and example_str != response):
                    with st.expander(f"💡 Example from {st.session_state.selected_example_persona}"):
                        st.write(f"*{example_str}*")
        
        st.markdown("---")
    
    # Navigation buttons at bottom
    st.markdown('<div class="navigation-buttons">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_page > 0:
            if st.button("← Previous Page", key="nav_prev_bottom"):
                st.session_state.current_page -= 1
                st.rerun()
    
    with col2:
        # Show completion status
        completed = len([q for q in questions if q in st.session_state.responses and st.session_state.responses[q].strip()])
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 8px;">
            <strong>Progress: {completed}/{len(questions)} questions completed</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.current_page < total_pages - 1:
            if st.button("Next Page →", key="nav_next_bottom"):
                st.session_state.current_page += 1
                st.rerun()
        else:
            if st.button("🎉 Complete Setup", type="primary", key="complete_setup"):
                if save_responses():
                    st.balloons()
                    st.success("Onboarding completed successfully! Your device is now personalized.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
        MND Device Onboarding System | All responses are saved locally and can be edited at any time
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()