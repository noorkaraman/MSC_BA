# Integrated Prompt Evaluation for MND Prototype
#This integrates directly with our existing prototype_core.ipynb

import os
import re
import pandas as pd
import time
import textstat
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import numpy as np

#Load environment variables
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

#Load existing data - change for relevant paths 
onboarding_df = pd.read_csv("/Users/noorkaraman/Desktop/ARP_files/23-08-2025-(2000)/data/Onboarding_Q&A.csv")
sarah_train = pd.read_csv("/Users/noorkaraman/Desktop/ARP_files/23-08-2025-(2000)/data/Sarah_Q&A.csv")
leo_train = pd.read_csv("/Users/noorkaraman/Desktop/ARP_files/23-08-2025-(2000)/data/Leo_Q&A.csv")
urja_train = pd.read_csv("/Users/noorkaraman/Desktop/ARP_files/23-08-2025-(2000)/data/Urja_Q&A.csv")

# ============== REUSE OUR EXISTING FUNCTIONS ==============
def get_persona_row(name: str) -> pd.Series:
    mask = onboarding_df["Persona"].astype(str).str.strip().str.casefold() == name.strip().casefold()
    if not mask.any():
        raise ValueError(f"Persona '{name}' not found in onboarding_df['Persona']")
    return onboarding_df.loc[mask].iloc[0]

def build_persona_text(row: pd.Series) -> str:
    lines = []
    for col in onboarding_df.columns:
        val = str(row.get(col, "")).strip()
        if val and val.lower() != "nan":
            lines.append(f"{col}: {val}")
    return "\n".join(lines)

def build_examples_blob(train_df: pd.DataFrame, setting: str) -> str:
    sub = train_df[
        train_df["Setting"].astype(str).str.strip().str.casefold() == setting.strip().casefold()
    ].dropna(subset=["Question", "Answer"])
    lines = [f"- Q: {q}\n  A: {a}" for q, a in zip(sub["Question"], sub["Answer"])]
    return f"Examples of past answers (Setting: {setting}, count={len(lines)}):\n" + "\n".join(lines)

def trim_to_char_budget(text: str, max_chars: int = 60000) -> str:
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

def get_train_df_for_persona(persona_name: str) -> pd.DataFrame:
    mapping = {
        "Sarah Ahmed": sarah_train,
        "Leonardo Carrey": leo_train,
        "Urja Mir": urja_train,
    }
    if persona_name not in mapping:
        raise ValueError(f"Unknown persona '{persona_name}'")
    return mapping[persona_name]

# ============== NEW EVALUATION CLASS ==============
class IntegratedPromptEvaluator:
    """Evaluate prompts integrated with your existing pipeline"""
    
    def __init__(self, persona_name: str = "Sarah Ahmed", setting: str = "Medical"):
        #Setup using your existing functions
        self.persona_name = persona_name
        self.setting = setting
        
        #Get persona data
        self.persona_row = get_persona_row(persona_name)
        self.persona_text = build_persona_text(self.persona_row)
        
        #Get training data
        self.train_df = get_train_df_for_persona(persona_name)
        self.examples_blob = build_examples_blob(self.train_df, setting)
        
        #Configure Gemini
        genai.configure(api_key=GEMINI_KEY)
        self.model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        
        self.results = []
    
    def get_all_prompts(self) -> Dict[str, Dict]:
        """Get all prompts to test with metadata"""
        
        prompts = {}
        
        #Our OG prompt 1
        prompts["original_basic"] = {
            "system": (
                "You are a personalised communication assistant for a user with Motor Neurone Disease.\n"
                f"The following describes their personal style, preferences, and context:\n{self.persona_text}\n"
                "Go through the entire prompt carefully to learn about the persona.\n"
                "When responding, strictly reflect their preferences in tone, style, humour, and avoid listed triggers.\n"
                "Return numbered options."
            ),
            "category": "original"
        }
        
        #Our original prompt 2
        prompts["original_organized"] = {
            "system": (
                "ROLE: Personalised communication assistant for a user with Motor Neurone Disease.\n"
                f"PERSONA: The following describes their personal style, preferences, and context \n{self.persona_text}\n"
                "GUIDELINES:\n"
                "• Go through the entire prompt carefully to learn about the persona.\n"
                "• You speak as the user in first person and produce sendable message options.\n"
                "• Strictly reflect their preferences in tone, style, humour, and avoid listed triggers\n"
                "• Return numbered options.\n"
                "OUTPUT STYLE:\n"
                "1) <option one>\n"
                "2) <option two>\n"
                "3) <option three>"
            ),
            "category": "original"
        }
        
        #Our original prompt 3
        prompts["original_strict"] = {
            "system": (
                "MND COMMUNICATION DEVICE – AS USER'S VOICE.\n"
                "ROLE: Personalised communication assistant for a user with Motor Neurone Disease.\n"
                f"PERSONA: The following describes their personal style, preferences, and context \n{self.persona_text}\n"
                "GUIDELINES:\n"
                "• Go through the entire prompt carefully to learn about the persona.\n"
                "• Speak as the user in first person and produce sendable message options addressed to the recipient.\n"
                "• Strictly reflect their preferences in tone, style, humour, and avoid listed triggers; use British spelling.\n"
                "• Use one - two sentences per option; aim for ≤ 30 words;\n"
                "• Mention the setting entity (e.g., clinic/team/recipient) if it improves clarity.\n"
                "• Provide 3 options that differ in phrasing but keep the same intent (no new facts).\n"
                "CONSISTENCY:\n"
                "• Do not invent details. Aim for a consistent tone/length/structure across runs for the same scenario.\n"
                "• Keep wording stable and clear; avoid unnecessary synonyms or filler.\n"
                "• Avoid explanations or meta commentary beyond the single sentence per option.\n"
                "PRIVACY & SAFETY:\n"
                "• Don't include personal/sensitive information unless clearly asked and available.\n"
                "OUTPUT:\n"
                "1) <option one>\n"
                "2) <option two>\n"
                "3) <option three>\n"
                "Return only the numbered options."
            ),
            "category": "original"
        }
        
        #Extract key persona traits for optimized prompts
        persona_lines = self.persona_text.split('\n')[:5]  #First 5 lines
        persona_summary = '\n'.join(persona_lines)
        
        #Optimized based on Google's strategies
        prompts["optimized_concise"] = {
            "system": (
                f"I'm {self.persona_name} with MND. Generate 3 response options.\n"
                "Rules: First person, <30 words each, British spelling.\n"
                f"My style: {persona_summary}\n"
                "Format: 1) 2) 3)"
            ),
            "category": "optimized"
        }
        
        prompts["optimized_structured"] = {
            "system": (
                f"Task: Generate 3 responses as {self.persona_name}.\n\n"
                f"Context:\n{persona_summary}\n\n"
                "Requirements:\n"
                "- First person\n"
                "- 15-30 words\n"
                "- Natural tone\n"
                "- Format: 1) response 2) response 3) response"
            ),
            "category": "optimized"
        }
        
        prompts["optimized_example_driven"] = {
            "system": (
                f"Generate responses as: {self.persona_name}\n"
                f"Style examples from past: {self.examples_blob[:200]}...\n\n"
                "Create 3 options (first person, <30 words each):\n"
                "1)\n2)\n3)"
            ),
            "category": "optimized"
        }
        
        prompts["optimized_minimal"] = {
            "system": (
                f"You are {self.persona_name}. Reply with 3 numbered options, 20 words max each."
            ),
            "category": "optimized"
        }
        
        return prompts
    
    def split_numbered_options(self, text: str, n: int = 3) -> List[str]:
        """Parse numbered responses - reusing your logic"""
        n = max(1, min(10, int(n)))
        
        if not isinstance(text, str) or not text.strip():
            return [""] * n
        
        #Use our existing regex pattern
        _num_pat = re.compile(r"^\s*(?:\d+[\).\-:]|\-\s*|\•\s*)\s*(.+?)$", re.MULTILINE)
        
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        opts = []
        for ln in lines:
            m = _num_pat.match(ln)
            opts.append(m.group(1).strip() if m else ln)
        
        if len([o for o in opts if o]) < n:
            chunks = re.split(r"(?:^|\s)(?:1\.|2\.|3\.|4\.|5\.)\\s*", text)
            chunks = [c.strip() for c in chunks if c.strip()]
            merged = []
            for x in opts + chunks:
                if x and x not in merged:
                    merged.append(x)
            opts = merged
        
        return (opts + [""] * n)[:n]
    
    def evaluate_single_prompt(self, prompt_name: str, system_prompt: str, 
                              question: str, sentiment: str = "neutral",
                              num_responses: int = 3, num_runs: int = 5) -> Dict:
        """Evaluate one prompt with multiple runs"""
        
        #Build user prompt using out format
        user_prompt = (
            f"Conversation setting: {self.setting}\n\n"
            f"{trim_to_char_budget(self.examples_blob, 60000)}\n\n"
            f"New question: {question}\n\n"
            f"Use a HARD '{sentiment}' sentiment in every option.\n"
            f"Provide exactly {num_responses} concise reply option(s), labelled 1, 2, 3... (no extra text)."
        )
        
        times = []
        readability_scores = []
        word_counts = []
        responses_collected = []
        
        for run in range(num_runs):
            try:
                start = time.time()
                
                #Generate using Gemini
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.model.generate_content(full_prompt)
                raw_text = response.text if hasattr(response, "text") else str(response)
                
                elapsed = time.time() - start
                times.append(elapsed)
                
                #Parse options
                options = self.split_numbered_options(raw_text, num_responses)
                responses_collected.append(options)
                
                #Calculate metrics for each option
                for opt in options:
                    if opt.strip():
                        #Readability
                        flesch = textstat.flesch_reading_ease(opt)
                        readability_scores.append(flesch)
                        
                        #Word count
                        words = len(opt.split())
                        word_counts.append(words)
                
            except Exception as e:
                print(f"  Error in run {run+1}: {e}")
                continue
        
        #Calculate stats
        return {
            "prompt_name": prompt_name,
            "question": question,
            "avg_time": np.mean(times) if times else 0,
            "std_time": np.std(times) if times else 0,
            "avg_readability": np.mean(readability_scores) if readability_scores else 0,
            "readability_interpretation": self.interpret_flesch(np.mean(readability_scores)) if readability_scores else "N/A",
            "avg_word_count": np.mean(word_counts) if word_counts else 0,
            "word_count_compliance": sum(15 <= w <= 30 for w in word_counts) / len(word_counts) * 100 if word_counts else 0,
            "successful_runs": len(times),
            "sample_responses": responses_collected[0] if responses_collected else [],
            "all_responses": responses_collected
        }
    
    def interpret_flesch(self, score: float) -> str:
        """Interpret Flesch score for accessibility"""
        if score >= 90:
            return "Very Easy (5th grade)"
        elif score >= 80:
            return "Easy (6th grade)"
        elif score >= 70:
            return "Fairly Easy (7th grade)"
        elif score >= 60:
            return "Standard (8-9th grade)"
        elif score >= 50:
            return "Fairly Difficult (10-12th)"
        elif score >= 30:
            return "Difficult (College)"
        else:
            return "Very Difficult (Graduate)"
    
    def run_comprehensive_evaluation(self, num_runs: int = 5) -> pd.DataFrame:
        """Run evaluation across all prompts"""
        
        #Test questions covering different scenarios
        test_questions = [
            "How are you feeling today?",
            "Do you need any help with that?",
            "Can we schedule a follow-up appointment?",
            "What would you like for lunch?",
            "Are you comfortable with the current medication?"
        ]
        
        all_prompts = self.get_all_prompts()
        results = []
        
        print(f"Evaluating {len(all_prompts)} prompts with {len(test_questions)} questions...")
        print(f"Persona: {self.persona_name}, Setting: {self.setting}")
        print("-" * 50)
        
        for q_idx, question in enumerate(test_questions):
            print(f"\nQuestion {q_idx+1}/{len(test_questions)}: '{question}'")
            
            for prompt_name, prompt_data in all_prompts.items():
                print(f"  Testing: {prompt_name}...")
                
                result = self.evaluate_single_prompt(
                    prompt_name=prompt_name,
                    system_prompt=prompt_data["system"],
                    question=question,
                    num_runs=num_runs
                )
                result["category"] = prompt_data["category"]
                results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_final_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive evaluation report"""
        
        report = "=" * 80 + "\n"
        report += "MND COMMUNICATION DEVICE - PROMPT EVALUATION REPORT\n"
        report += f"Persona: {self.persona_name} | Setting: {self.setting}\n"
        report += "=" * 80 + "\n\n"
        
        #Overall stats by prompt
        prompt_stats = df.groupby('prompt_name').agg({
            'avg_time': 'mean',
            'avg_readability': 'mean',
            'avg_word_count': 'mean',
            'word_count_compliance': 'mean',
            'successful_runs': 'sum'
        }).round(2)
        
        report += "PERFORMANCE METRICS BY PROMPT:\n"
        report += "-" * 40 + "\n"
        report += prompt_stats.to_string() + "\n\n"
        
        #Category comparison
        category_stats = df.groupby('category').agg({
            'avg_time': 'mean',
            'avg_readability': 'mean',
            'word_count_compliance': 'mean'
        }).round(2)
        
        report += "ORIGINAL vs OPTIMIZED:\n"
        report += "-" * 40 + "\n"
        report += category_stats.to_string() + "\n\n"
        
        #Winners
        report += "BEST PERFORMERS:\n"
        report += "-" * 40 + "\n"
        report += f"⚡ Fastest: {prompt_stats['avg_time'].idxmin()} ({prompt_stats['avg_time'].min():.2f}s)\n"
        report += f"Most Readable: {prompt_stats['avg_readability'].idxmax()} (Score: {prompt_stats['avg_readability'].max():.1f})\n"
        report += f"Best Word Count Compliance: {prompt_stats['word_count_compliance'].idxmax()} ({prompt_stats['word_count_compliance'].max():.1f}%)\n\n"
        
        #Recommended prompt
        #Weight: 40% speed, 40% readability, 20% compliance
        prompt_stats['composite_score'] = (
            (1 / prompt_stats['avg_time']) * 0.4 +  #Lower time is better
            prompt_stats['avg_readability'] / 100 * 0.4 +  #Higher readability is better
            prompt_stats['word_count_compliance'] / 100 * 0.2  #Higher compliance is better
        )
        
        best_prompt = prompt_stats['composite_score'].idxmax()
        
        report += "RECOMMENDED PROMPT:\n"
        report += "-" * 40 + "\n"
        report += f"Based on composite scoring: {best_prompt}\n"
        report += f"Composite Score: {prompt_stats.loc[best_prompt, 'composite_score']:.3f}\n\n"
        
        #Sample outputs from best prompt
        best_samples = df[df['prompt_name'] == best_prompt].iloc[0]
        report += f"Sample outputs from {best_prompt}:\n"
        report += f"Question: {best_samples['question']}\n"
        for i, resp in enumerate(best_samples['sample_responses'], 1):
            if resp:
                report += f"  {i}) {resp}\n"
        
        return report

#Main execution function
def run_integrated_evaluation():
    """Main function to run the complete evaluation"""
    
    print("Starting Integrated Prompt Evaluation for MND Device")
    print("=" * 50)
    
    #Test with diff personas and settings
    test_configs = [
        {"persona": "Sarah Ahmed", "setting": "Medical"},
        {"persona": "Sarah Ahmed", "setting": "Family & Friends"},
        {"persona": "Leonardo Carrey", "setting": "Work"},
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n\nTesting: {config['persona']} in {config['setting']} setting")
        print("-" * 50)
        
        evaluator = IntegratedPromptEvaluator(
            persona_name=config["persona"],
            setting=config["setting"]
        )
        
        #Run eval
        results_df = evaluator.run_comprehensive_evaluation(num_runs=3)  # Reduce for faster testing
        results_df['test_config'] = f"{config['persona']}_{config['setting']}"
        all_results.append(results_df)
        
        #Generate report for this specific configuration
        report = evaluator.generate_final_report(results_df)
        
        #Save individual reports
        filename = f"prompt_eval_{config['persona'].replace(' ', '_')}_{config['setting'].replace(' & ', '_')}.txt"
        with open(filename, "w") as f:
            f.write(report)
        
        print(f"\nReport saved to {filename}")
        print("\n" + "=" * 30 + " SUMMARY " + "=" * 30)
        print(report.split("RECOMMENDED PROMPT:")[1].split("\n\n")[0])
    
    #Combine all results into a csv 
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv("complete_prompt_evaluation.csv", index=False)
    
    print("\n" + "=" * 50)
    print("Evaluation complete! Check the generated files for detailed results.")
    
    return combined_df

if __name__ == "__main__":
    results = run_integrated_evaluation()