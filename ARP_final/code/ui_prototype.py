import streamlit as st
import pandas as pd
import re
import os
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="MND Communication Device",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .block-container { 
        padding-top: 1rem; 
        padding-bottom: 2rem; 
        max-width: 100%;
    }
    .buffer-box {
        width: 100%; 
        min-height: 80px;
        border-radius: 16px; 
        border: 2px solid #e5e7eb;
        padding: 20px; 
        font-size: 24px; 
        background: #f9fafb;
        text-align: center;
        margin: 10px 0;
        font-weight: 500;
    }
    .incoming-question-display {
        background: #f0f9ff;
        border: 2px solid #0ea5e9;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        font-size: 18px;
        font-weight: 500;
        color: #0c4a6e;
    }
    .section-header {
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #374151;
        margin: 15px 0 10px 0;
    }
    .response-time {
        text-align: center;
        color: #6b7280;
        font-size: 12px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)



# ============== CORE HELPER FUNCTIONS ==============
def get_persona_row(name: str, onboarding_df: pd.DataFrame) -> pd.Series:
    """Fetch persona's onboarding data by name"""
    mask = onboarding_df["Persona"].astype(str).str.strip().str.casefold() == name.strip().casefold()
    if not mask.any():
        raise ValueError(f"Persona '{name}' not found")
    return onboarding_df.loc[mask].iloc[0]

def build_persona_text(row: pd.Series, onboarding_df: pd.DataFrame) -> str:
    """Build persona text block from onboarding data"""
    lines = []
    for col in onboarding_df.columns:
        val = str(row.get(col, "")).strip()
        if val and val.lower() != "nan":
            lines.append(f"{col}: {val}")
    return "\n".join(lines)

def build_examples_blob(train_df: pd.DataFrame, setting: str) -> str:
    """Build examples blob for a specific setting"""
    sub = train_df[
        train_df["Setting"].astype(str).str.strip().str.casefold() == setting.strip().casefold()
    ].dropna(subset=["Question", "Answer"])
    lines = [f"- Q: {q}\n  A: {a}" for q, a in zip(sub["Question"], sub["Answer"])]
    return f"Examples of past answers (Setting: {setting}, count={len(lines)}):\n" + "\n".join(lines)

def trim_to_char_budget(text: str, max_chars: int = 60000) -> str:
    """Trim text to fit within character budget"""
    if len(text) <= max_chars:
        return text
    out, total = [], 0
    for ln in text.splitlines():
        L = len(ln) + 1
        if total + L > max_chars:
            break
        out.append(ln)
        total += L
    out.append("\n...[trimmed due to context budget]...")
    return "\n".join(out)

def split_numbered_options(text: str, n: int):
    """Split LLM output into exactly n options"""
    n = max(1, min(10, int(n)))
    
    if not isinstance(text, str) or not text.strip():
        return [""] * n
    
    # Regex pattern for numbered/bulleted lines
    _num_pat = re.compile(r"^\s*(?:\d+[\).\-:]|\-\s*|•\s*)\s*(.+?)$", re.MULTILINE)
    
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    opts = []
    for ln in lines:
        m = _num_pat.match(ln)
        opts.append(m.group(1).strip() if m else ln)
    
    # Fallback: if too few, split by common numbering tokens
    if len([o for o in opts if o]) < n:
        chunks = re.split(r"(?:^|\s)(?:1\.|2\.|3\.|4\.|5\.|6\.|7\.|8\.|9\.|10\.)\s*", text)
        chunks = [c.strip() for c in chunks if c.strip()]
        merged = []
        for x in opts + chunks:
            if x and x not in merged:
                merged.append(x)
        opts = merged
    
    return (opts + [""] * n)[:n]

def get_optimized_system_prompt(persona_text: str, num_responses: int, stm_prompt: int = 2) -> str:
    """Get the optimized system prompt based on evaluation results"""
    
    # Based on your stm_prompt preference (2 = organized)
    if stm_prompt == 2:
        # This is the organized prompt from your prototype
        return (
            "ROLE: Personalised communication assistant for a user with Motor Neurone Disease.\n"
            f"PERSONA: The following describes their personal style, preferences, and context \n{persona_text}\n"
            "GUIDELINES:\n"
            "• Go through the entire prompt carefully to learn about the persona.\n"
            "• You speak as the user in first person and produce sendable message options.\n"
            "• Strictly reflect their preferences in tone, style, humour, and avoid listed triggers\n"
            "• Return numbered options.\n"
            "OUTPUT STYLE:\n"
            "1) <option one>\n"
            "2) <option two>\n"
            "3) <option three>\n"
            "...\n"
            f"n) <option {num_responses}>"
        )
    else:
        # Optimized concise prompt (if evaluation shows it's better)
        persona_lines = persona_text.split('\n')[:5]
        persona_summary = '\n'.join(persona_lines)
        return (
            f"Task: Generate {num_responses} responses as the user.\n\n"
            f"Context:\n{persona_summary}\n\n"
            "Requirements:\n"
            "- First person voice\n"
            "- 15-30 words each\n"
            "- Natural, conversational tone\n"
            "- British spelling\n"
            f"- Format: 1) response 2) response ... {num_responses}) response"
        )

def generate_llm_responses(persona_text: str, setting: str, question: str, 
                          examples_blob: str, num_responses: int = 3, 
                          sentiment: str = "neutral", temperature: float = 1.0,
                          stm_prompt: int = 2) -> tuple:
    """Generate responses using Gemini 2.5 Flash"""
    
    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")
    
    genai.configure(api_key=api_key)
    
    # Use Gemini 2.5 Flash as determined by evaluation
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        generation_config={
            'temperature': temperature,
            'max_output_tokens': 6000,
        }
    )
    
    # Get optimized system prompt
    system_prompt = get_optimized_system_prompt(persona_text, num_responses, stm_prompt)
    
    # Build user prompt (same as prototype)
    examples_blob = trim_to_char_budget(examples_blob, 60000)
    user_prompt = (
        f"Conversation setting: {setting}\n\n"
        f"{examples_blob}\n\n"
        f"New question: {question}\n\n"
        f"Use a HARD '{sentiment}' sentiment in every option.\n"
        f"Provide exactly {num_responses} concise reply option(s), labelled 1, 2, 3... (no extra text)."
    )
    
    # Combine prompts
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    # Generate with timing
    start_time = time.time()
    try:
        response = model.generate_content(full_prompt)
        raw_text = response.text if hasattr(response, "text") else str(response)
        elapsed_time = time.time() - start_time
        return raw_text, elapsed_time
    except Exception as e:
        raise Exception(f"Gemini API Error: {str(e)}")

# ============== DATA LOADING ==============
@st.cache_data
def load_data():
    """Load all required CSV files"""
    try:
        data = {}
        possible_paths = ["data/", "", "../data/"]
        files_to_load = {
            'onboarding': 'Onboarding_QnA.csv',
            'sarah': 'Sarah_QnA.csv',
            'leo': 'Leo_QnA.csv',
            'urja': 'Urja_QnA.csv'
        }
        for key, filename in files_to_load.items():
            file_found = False
            for path in possible_paths:
                try:
                    data[key] = pd.read_csv(path + filename)
                    file_found = True
                    break
                except FileNotFoundError:
                    continue
            if not file_found:
                st.error(f"Could not find {filename}")
                return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ============== SESSION STATE ==============
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'text_buffer': "",
        'llm_mode': False,
        'llm_suggestions': [],
        'selected_persona': "Sarah Ahmed",
        'setting': "Family & Friends",
        'incoming_question': "Hi, how are you?",
        'num_responses': 3,
        'sentiment': "neutral",
        'temperature': 1.0,
        'stm_prompt': 2,  # Default to organized prompt
        'last_generation_time': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============== UI HELPER FUNCTIONS ==============
def add_word(w):
    """Add word to text buffer"""
    if not st.session_state.text_buffer or st.session_state.text_buffer.endswith(" "):
        st.session_state.text_buffer += w
    else:
        st.session_state.text_buffer += " " + w

def add_char(c):
    """Add character to text buffer"""
    st.session_state.text_buffer += c

def del_word():
    """Delete last word from text buffer"""
    txt = st.session_state.text_buffer.rstrip()
    if not txt:
        return
    parts = txt.split(" ")
    if len(parts) > 1:
        st.session_state.text_buffer = " ".join(parts[:-1]) + " "
    else:
        st.session_state.text_buffer = ""

def clear_all():
    """Clear text buffer"""
    st.session_state.text_buffer = ""

def process_llm_generation():
    """Process LLM response generation"""
    data = load_data()
    if not data:
        st.warning("Could not load training data")
        return
    
    question = st.session_state.incoming_question
    if not question.strip():
        st.warning("Please enter a question to respond to")
        return
    
    try:
        with st.spinner("Generating personalized responses..."):
            # Get persona data
            persona_row = get_persona_row(st.session_state.selected_persona, data['onboarding'])
            persona_text = build_persona_text(persona_row, data['onboarding'])
            
            # Map persona to training data
            persona_mapping = {
                "Sarah Ahmed": data['sarah'],
                "Leonardo Carrey": data['leo'],
                "Urja Mir": data['urja']
            }
            
            train_df = persona_mapping.get(st.session_state.selected_persona)
            if train_df is None:
                st.error(f"Training data not found for {st.session_state.selected_persona}")
                return
            
            # Build examples
            examples_blob = build_examples_blob(train_df, st.session_state.setting)
            
            # Generate responses
            raw_response, elapsed_time = generate_llm_responses(
                persona_text=persona_text,
                setting=st.session_state.setting,
                question=question,
                examples_blob=examples_blob,
                num_responses=st.session_state.num_responses,
                sentiment=st.session_state.sentiment,
                temperature=st.session_state.temperature,
                stm_prompt=st.session_state.stm_prompt
            )
            
            if raw_response:
                options = split_numbered_options(raw_response, st.session_state.num_responses)
                valid_options = [opt for opt in options if opt.strip()]
                
                if valid_options:
                    st.session_state.llm_suggestions = valid_options
                    st.session_state.last_generation_time = elapsed_time
                    st.success(f"Generated {len(valid_options)} responses in {elapsed_time:.2f}s")
                else:
                    st.warning("No valid responses generated. Please try again.")
            else:
                st.error("Failed to generate responses.")
                
    except Exception as e:
        st.error(f"Error generating responses: {str(e)}")

# ============== MAIN APPLICATION ==============
def main():
    initialize_session_state()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("📱 Device Settings")
        
        # Incoming Question
        st.markdown("### 💬 Incoming Question")
        st.session_state.incoming_question = st.text_area(
            "Question to respond to:",
            value=st.session_state.incoming_question,
            height=100,
            help="Enter the question you want to respond to"
        )
        
        st.markdown("---")
        
        # Persona Settings
        st.markdown("### 👤 Persona Settings")
        st.session_state.selected_persona = st.selectbox(
            "Active Persona:", 
            ["Sarah Ahmed", "Leonardo Carrey", "Urja Mir"],
            help="Select the persona whose communication style to use"
        )
        
        st.session_state.setting = st.selectbox(
            "Conversation Setting:", 
            ["Medical", "Work", "Family & Friends"],
            help="Choose the appropriate context for the conversation"
        )
        
        st.markdown("---")
        
        # Response Configuration
        st.markdown("### ⚙️ Response Configuration")
        
        st.session_state.num_responses = st.slider(
            "Number of Responses:",
            min_value=1,
            max_value=5,
            value=st.session_state.num_responses,
            help="Number of response options to generate"
        )
        
        st.session_state.sentiment = st.selectbox(
            "Response Sentiment:",
            ["neutral", "positive", "negative", "formal", "casual"],
            help="Emotional tone for the responses"
        )
        
        # Advanced settings (collapsible)
        with st.expander("Advanced Settings"):
            st.session_state.temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Controls randomness (0=deterministic, 2=very creative)"
            )
            
            st.session_state.stm_prompt = st.radio(
                "Prompt Style:",
                options=[2, 1],
                format_func=lambda x: "Organized (Recommended)" if x == 2 else "Simple",
                help="System prompt complexity"
            )
        
        # Generate button
        if st.button("🚀 Generate Responses", type="primary", use_container_width=True):
            process_llm_generation()
        
        # Display generation time if available
        if st.session_state.last_generation_time:
            st.markdown(f"<div class='response-time'>Last generation: {st.session_state.last_generation_time:.2f}s</div>", 
                       unsafe_allow_html=True)
    
    # Main Interface
    # Display incoming question
    st.markdown(f"""
        <div class="incoming-question-display">
            <strong>Responding to:</strong> {st.session_state.incoming_question}
        </div>
    """, unsafe_allow_html=True)
    
    # Text buffer display
    buffer_content = st.session_state.text_buffer if st.session_state.text_buffer else "Type your response..."
    st.markdown(f"<div class='buffer-box'>{buffer_content}</div>", unsafe_allow_html=True)
    
    # Core words or LLM responses section
    if not st.session_state.llm_mode:
        st.markdown("<div class='section-header'>CORE WORDS</div>", unsafe_allow_html=True)
        core_words = [
            ["aid", "any", "have", "of", "one", "make", "need", "say", "take", "the", "want"],
            ["all", "are", "he/her", "off", "or", "may", "no", "should", "tell", "then", "was"],
            ["also", "as", "help", "okay", "other", "maybe", "not", "some", "thank", "there", "when"],
            ["and", "at", "how", "on", "our", "might", "now", "stop", "that", "these", "where"]
        ]
        for row in core_words:
            cols = st.columns(len(row))
            for i, word in enumerate(row):
                with cols[i]:
                    if st.button(word, key=f"core_{word}", use_container_width=True):
                        add_word(word)
    else:
        st.markdown("<div class='section-header'>AI-GENERATED RESPONSES</div>", unsafe_allow_html=True)
        if st.session_state.llm_suggestions:
            # Display responses in columns (max 5 per row)
            num_suggestions = len(st.session_state.llm_suggestions)
            if num_suggestions <= 5:
                cols = st.columns(num_suggestions)
                for i, suggestion in enumerate(st.session_state.llm_suggestions):
                    with cols[i]:
                        if st.button(suggestion, key=f"llm_{i}", use_container_width=True):
                            st.session_state.text_buffer = suggestion
            else:
                # Multiple rows for more than 5 suggestions
                for row_start in range(0, num_suggestions, 5):
                    row_end = min(row_start + 5, num_suggestions)
                    cols = st.columns(row_end - row_start)
                    for i, suggestion in enumerate(st.session_state.llm_suggestions[row_start:row_end]):
                        with cols[i]:
                            if st.button(suggestion, key=f"llm_{row_start + i}", use_container_width=True):
                                st.session_state.text_buffer = suggestion
        else:
            st.info("Click 'Generate Responses' in the sidebar to get AI suggestions")
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    # Keyboard Layout
    left_col, keyboard_col, right_col = st.columns([1, 6, 1])
    
    # Left navigation
    with left_col:
        nav_items = [
            ("👥", "People"),
            ("📍", "Location"),
            ("📦", "Objects"),
            ("📁", "Files"),
            ("🏠", "Home"),
            ("←", "Back")
        ]
        for emoji, tooltip in nav_items:
            st.button(emoji, key=f"nav_{emoji}", help=tooltip, use_container_width=True)
    
    # QWERTY Keyboard
    with keyboard_col:
        # Row 1: QWERTYUIOP
        row1 = list("qwertyuiop")
        cols = st.columns(len(row1))
        for i, key in enumerate(row1):
            with cols[i]:
                if st.button(key.upper(), key=f"k1_{i}", use_container_width=True):
                    add_char(key)
        
        # Row 2: ASDFGHJKL (with offset)
        row2 = list("asdfghjkl")
        cols = st.columns([0.5] + [1]*9 + [0.5])
        for i, key in enumerate(row2):
            with cols[i+1]:
                if st.button(key.upper(), key=f"k2_{i}", use_container_width=True):
                    add_char(key)
        
        # Row 3: ZXCVBNM (with more offset)
        row3 = list("zxcvbnm")
        cols = st.columns([1.5] + [1]*7 + [1.5])
        for i, key in enumerate(row3):
            with cols[i+1]:
                if st.button(key.upper(), key=f"k3_{i}", use_container_width=True):
                    add_char(key)
        
        # Spacebar
        if st.button("SPACE", key="spacebar", use_container_width=True):
            add_char(" ")
    
    # Right action buttons
    with right_col:
        st.button("❌", key="del", help="Delete word", use_container_width=True, on_click=del_word)
        st.button("✏️", key="edit", help="Edit", use_container_width=True)
        st.button("🗑️", key="clear", help="Clear all", use_container_width=True, on_click=clear_all)
        st.button("⚙️", key="settings", help="Settings", use_container_width=True)
        st.button("📤", key="share", help="Share", use_container_width=True)
        
        # LLM Mode Toggle
        toggle_emoji = "🤖" if st.session_state.llm_mode else "💭"
        toggle_help = "Switch to Core Words" if st.session_state.llm_mode else "Switch to AI Responses"
        if st.button(toggle_emoji, key="llm_toggle", help=toggle_help, use_container_width=True):
            st.session_state.llm_mode = not st.session_state.llm_mode
            if st.session_state.llm_mode and st.session_state.incoming_question.strip():
                process_llm_generation()
            st.rerun()

if __name__ == "__main__":
    main()