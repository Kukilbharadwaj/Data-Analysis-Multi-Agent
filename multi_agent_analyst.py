"""
Multi-Agent AI Data Analyst Platform
A simple implementation using LangGraph and Groq
Enhanced with advanced data loading for complex and unstructured data
"""

import os
import pandas as pd
from typing import TypedDict, Annotated, List, AsyncGenerator, Dict, Any
from groq import Groq
from langgraph.graph import StateGraph, END
import json
import asyncio
from dotenv import load_dotenv
from data_loader import load_and_process_data

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables. "
        "Please set it in your .env file or environment."
    )
client = Groq(api_key=groq_api_key)

# Define the state that will be passed between agents
class AnalysisState(TypedDict):
    dataset_path: str
    data_summary: str
    quality_report: dict  # Added for data quality information
    analysis_plan: str
    exploration_results: str
    sql_queries: str
    insights: str
    validation_results: str
    final_report: str
    messages: List[str]

def call_llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Helper function to call Groq LLM"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# AGENT 1: Planner Agent
def planner_agent(state: AnalysisState) -> AnalysisState:
    """Creates an analysis plan based on the dataset"""
    print("\n🎯 Planner Agent: Creating analysis plan...")
    
    prompt = f"""You are a Data Analysis Planner. Based on this dataset summary:
{state['data_summary']}

Create a structured analysis plan. Include:
1. Key metrics to analyze
2. Patterns to look for
3. Data quality checks needed
4. Visualization recommendations

Keep it concise and actionable."""
    
    analysis_plan = call_llm(prompt)
    state['analysis_plan'] = analysis_plan
    state['messages'].append("✅ Planner: Analysis plan created")
    print(f"Plan created: {len(analysis_plan)} characters")
    return state

# AGENT 2: Explorer Agent
def explorer_agent(state: AnalysisState) -> AnalysisState:
    """Profiles and explores the data with advanced processing"""
    print("\n🔍 Explorer Agent: Profiling data...")
    
    # Load and profile the actual data using advanced loader
    try:
        # Use advanced data loader
        df, quality_report = load_and_process_data(state['dataset_path'])
        
        # Build comprehensive data profile
        data_profile = f"""
Dataset Profile:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column Names: {', '.join(df.columns.tolist())}
- Data Types: {df.dtypes.to_dict()}
- Missing Values (after processing): {df.isnull().sum().to_dict()}

Data Quality Report:
- Original Encoding: {quality_report['file_info'].get('encoding', 'N/A')}
- Duplicates Removed: {quality_report['data_quality'].get('duplicates_removed', 0)}
- Columns with Missing Data: {len(quality_report['data_quality'].get('missing_values', {}))}
- Outliers Detected: {len(quality_report['data_quality'].get('outliers', {}))} columns

Numeric Summary:
{df.describe().to_string() if len(df.select_dtypes(include='number').columns) > 0 else 'No numeric columns'}

Sample Data (first 5 rows):
{df.head(5).to_string()}

Categorical Insights:
{json.dumps(quality_report['statistics'].get('categorical', {}), indent=2)[:500] if quality_report['statistics'].get('categorical') else 'No categorical columns'}
"""
        
        prompt = f"""You are a Data Explorer analyzing processed and cleaned data. This data has been through advanced quality checks:
{data_profile}

Provide:
1. Data quality observations (note any issues found and handled)
2. Interesting patterns or anomalies in the data
3. Key characteristics of the dataset
4. Recommendations for further analysis

Be specific about the data cleaning that was performed and what it means for the analysis."""
        
        exploration_results = call_llm(prompt)
        state['exploration_results'] = data_profile + "\n\nExpert Analysis:\n" + exploration_results
        state['quality_report'] = quality_report
        state['messages'].append("✅ Explorer: Advanced data profiling complete")
        print("Data exploration complete with quality report")
        
    except Exception as e:
        state['exploration_results'] = f"Error exploring data: {str(e)}"
        state['quality_report'] = {}
        state['messages'].append("⚠️ Explorer: Error occurred")
    
    return state

# AGENT 3: SQL Agent
def sql_agent(state: AnalysisState) -> AnalysisState:
    """Generates SQL-like queries and performs data operations"""
    print("\n💾 SQL Agent: Generating queries...")
    
    prompt = f"""You are a SQL Query Specialist. Based on this analysis plan:
{state['analysis_plan']}

And this data exploration:
{state['exploration_results'][:500]}...

Generate 3-5 pandas query/operation suggestions that would extract valuable insights.
Format as Python pandas operations (e.g., df.groupby(), df.query(), etc.)"""
    
    sql_queries = call_llm(prompt)
    state['sql_queries'] = sql_queries
    state['messages'].append("✅ SQL Agent: Queries generated")
    print("Query suggestions created")
    return state

# AGENT 4: Insight Agent
def insight_agent(state: AnalysisState) -> AnalysisState:
    """Finds patterns and generates insights with advanced analytics"""
    print("\n💡 Insight Agent: Finding patterns...")
    
    # Actually run some basic analysis
    try:
        # Load processed data
        df, _ = load_and_process_data(state['dataset_path'])
        
        # Build advanced analytics summary
        analytics_summary = ""
        
        # Numeric column analysis
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            analytics_summary += f"Correlation Matrix:\n{corr_matrix.to_string()}\n\n"
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_corr.append(
                            f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]}: {corr_val:.3f}"
                        )
            
            if strong_corr:
                analytics_summary += "Strong Correlations (>0.5):\n" + "\n".join(strong_corr) + "\n\n"
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        if categorical_cols:
            analytics_summary += "Categorical Distributions:\n"
            for col in categorical_cols[:5]:  # Limit to first 5
                value_counts = df[col].value_counts().head(3)
                analytics_summary += f"  {col}: {dict(value_counts)}\n"
        
        # Data quality from quality report
        if state.get('quality_report'):
            qr = state['quality_report']
            analytics_summary += f"\nData Quality Insights:\n"
            analytics_summary += f"- Duplicates removed: {qr['data_quality'].get('duplicates_removed', 0)}\n"
            if qr['data_quality'].get('outliers'):
                analytics_summary += f"- Outliers detected in {len(qr['data_quality']['outliers'])} columns\n"
        
        prompt = f"""You are an Advanced Insight Analyst. Based on:

Analysis Plan:
{state['analysis_plan'][:400]}...

Exploration Results:
{state['exploration_results'][:400]}...

Advanced Analytics:
{analytics_summary}

Identify 5-7 KEY INSIGHTS and patterns in the data. Focus on:
1. Statistical relationships and correlations
2. Distribution patterns and anomalies
3. Business-relevant trends
4. Data quality insights and their implications
5. Actionable patterns

Be specific, quantitative where possible, and actionable."""
        
        insights = call_llm(prompt)
        state['insights'] = insights
        state['messages'].append("✅ Insight Agent: Advanced patterns identified")
        print("Insights generated with advanced analytics")
        
    except Exception as e:
        state['insights'] = f"Error generating insights: {str(e)}"
        state['messages'].append("⚠️ Insight Agent: Error occurred")
    
    return state

# AGENT 5: Validator Agent
def validator_agent(state: AnalysisState) -> AnalysisState:
    """Validates results and checks for errors"""
    print("\n✓ Validator Agent: Verifying results...")
    
    prompt = f"""You are a Data Validation Specialist. Review these insights:
{state['insights']}

Check for:
1. Logical consistency
2. Statistical validity
3. Potential biases or errors
4. Confidence level in findings

Provide a validation report."""
    
    validation_results = call_llm(prompt)
    state['validation_results'] = validation_results
    state['messages'].append("✅ Validator: Results verified")
    print("Validation complete")
    return state

# AGENT 6: Narrator Agent
def narrator_agent(state: AnalysisState) -> AnalysisState:
    """Creates final narrative report"""
    print("\n📝 Narrator Agent: Creating final report...")
    
    prompt = f"""You are a Data Storyteller creating a concise, user-friendly report for business users.

ANALYSIS PLAN:
{state['analysis_plan']}

KEY INSIGHTS:
{state['insights']}

VALIDATION:
{state['validation_results']}

Create a brief, actionable report in this EXACT format:

KEY FINDINGS:
- [Finding 1: One clear, specific insight]
- [Finding 2: One clear, specific insight]
- [Finding 3: One clear, specific insight]
- [Finding 4: One clear, specific insight]
- [Finding 5: One clear, specific insight]

RECOMMENDATIONS:
- [Action 1: Clear next step]
- [Action 2: Clear next step]
- [Action 3: Clear next step]
- [Action 4: Clear next step]

Keep each bullet point to 1-2 sentences. Be concrete and actionable."""
    
    final_report = call_llm(prompt)
    state['final_report'] = final_report
    state['messages'].append("✅ Narrator: Final report ready")
    print("Final report complete")
    return state


def generate_dashboard_data(df: pd.DataFrame, quality_report: dict) -> dict:
    """Generate dashboard-ready chart data for visualization"""
    import numpy as np
    
    dashboard = {
        "charts": [],
        "metrics": [],
        "data_quality_viz": {}
    }
    
    # Helper function to convert to native Python types
    def to_native(val):
        """Convert numpy/pandas types to native Python types"""
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        elif isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        elif pd.isna(val):
            return None
        return val
    
    # 1. Key Metrics Cards
    dashboard["metrics"] = [
        {"label": "Total Rows", "value": f"{len(df):,}", "icon": "database", "color": "blue"},
        {"label": "Columns", "value": int(len(df.columns)), "icon": "columns", "color": "green"},
        {"label": "Duplicates Removed", "value": int(quality_report['data_quality'].get('duplicates_removed', 0)), "icon": "trash", "color": "yellow"},
        {"label": "Data Quality", "value": "Good" if quality_report['data_quality'].get('duplicates_removed', 0) < len(df) * 0.1 else "Fair", "icon": "check-circle", "color": "purple"}
    ]
    
    # 2. Column Data Types Distribution (Pie Chart)
    type_counts = df.dtypes.value_counts().to_dict()
    type_names = [str(k) for k in type_counts.keys()]
    type_values = list(type_counts.values())
    
    dashboard["charts"].append({
        "id": "dataTypesPie",
        "type": "pie",
        "title": "Column Data Types Distribution",
        "labels": type_names,
        "data": type_values,
        "colors": ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
    })
    
    # 3. Missing Values Bar Chart
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0].head(10)
    
    if len(missing_cols) > 0:
        dashboard["charts"].append({
            "id": "missingValuesBar",
            "type": "bar",
            "title": "Missing Values by Column",
            "labels": [str(x) for x in missing_cols.index.tolist()],
            "data": [int(x) for x in missing_cols.values.tolist()],
            "color": "#EF4444"
        })
    
    # 4. Numeric Columns Distribution (for first 3 numeric columns)
    numeric_cols = df.select_dtypes(include=['number']).columns[:3]
    
    for idx, col in enumerate(numeric_cols):
        try:
            # Create histogram data
            hist_data, bin_edges = pd.cut(df[col].dropna(), bins=10, retbins=True, duplicates='drop')
            value_counts = hist_data.value_counts().sort_index()
            
            # Create labels from bin edges
            labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(value_counts))]
            
            dashboard["charts"].append({
                "id": f"distribution{idx}",
                "type": "bar",
                "title": f"Distribution of {col}",
                "labels": labels,
                "data": [int(x) for x in value_counts.values.tolist()],
                "color": ["#3B82F6", "#10B981", "#F59E0B"][idx]
            })
        except:
            pass  # Skip if histogram creation fails
    
    # 5. Top Values for Categorical Columns (first 2)
    categorical_cols = df.select_dtypes(include=['object']).columns[:2]
    
    for idx, col in enumerate(categorical_cols):
        try:
            top_values = df[col].value_counts().head(8)
            
            dashboard["charts"].append({
                "id": f"topValues{idx}",
                "type": "horizontalBar",
                "title": f"Top Values in {col}",
                "labels": [str(x) for x in top_values.index.tolist()],
                "data": [int(x) for x in top_values.values.tolist()],
                "color": ["#8B5CF6", "#EC4899"][idx]
            })
        except:
            pass
    
    # 6. Data Quality Summary (Doughnut Chart)
    total_cells = len(df) * len(df.columns)
    missing_cells = int(df.isnull().sum().sum())
    filled_cells = int(total_cells - missing_cells)
    
    dashboard["data_quality_viz"] = {
        "id": "dataQualityDoughnut",
        "type": "doughnut",
        "title": "Overall Data Completeness",
        "labels": ["Complete Data", "Missing Data"],
        "data": [filled_cells, missing_cells],
        "colors": ["#10B981", "#EF4444"]
    }
    
    # 7. Correlation Matrix (if we have numeric columns)
    numeric_df = df.select_dtypes(include=['number'])
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr()
        # Get top 5 correlations (excluding diagonal)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = float(abs(corr_matrix.iloc[i, j]))
                corr_pairs.append({
                    "pair": f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                    "correlation": corr_val
                })
        
        # Sort by correlation strength
        corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)
        top_correlations = corr_pairs[:5]
        
        if top_correlations:
            dashboard["charts"].append({
                "id": "correlationBar",
                "type": "bar",
                "title": "Top Column Correlations",
                "labels": [str(p["pair"]) for p in top_correlations],
                "data": [float(p["correlation"]) for p in top_correlations],
                "color": "#06B6D4"
            })
    
    return dashboard


def extract_user_friendly_summary(state: AnalysisState, dataset_path: str) -> dict:
    """Extract clean, user-friendly summary from analysis results"""
    
    # Get dataset info
    try:
        df = pd.read_csv(dataset_path)
        dataset_summary = f"{len(df):,} rows × {len(df.columns)} columns"
    except:
        dataset_summary = "Dataset analyzed"
    
    # Parse the final report to extract findings and recommendations
    final_report = state['final_report']
    
    key_findings = []
    recommendations = []
    
    # Simple parsing logic
    lines = final_report.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if 'KEY FINDINGS' in line.upper():
            current_section = 'findings'
        elif 'RECOMMENDATIONS' in line.upper():
            current_section = 'recommendations'
        elif line.startswith('-') or line.startswith('•'):
            # Remove bullet point and clean up
            clean_line = line.lstrip('- •').strip()
            if clean_line and len(clean_line) > 10:  # Filter out very short lines
                if current_section == 'findings' and len(key_findings) < 5:
                    key_findings.append(clean_line)
                elif current_section == 'recommendations' and len(recommendations) < 4:
                    recommendations.append(clean_line)
    
    # Fallback if parsing didn't work well
    if len(key_findings) < 3:
        key_findings = [
            "Data analysis completed successfully",
            "Insights extracted from your dataset",
            "Patterns and trends identified"
        ]
    
    if len(recommendations) < 2:
        recommendations = [
            "Review the identified patterns for business opportunities",
            "Consider further analysis on key metrics"
        ]
    
    return {
        "dataset_summary": dataset_summary,
        "key_findings": key_findings,
        "recommendations": recommendations
    }

# Build the LangGraph workflow
def create_workflow():
    """Creates the multi-agent workflow using LangGraph"""
    
    workflow = StateGraph(AnalysisState)
    
    # Add all agents as nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("explorer", explorer_agent)
    workflow.add_node("sql", sql_agent)
    workflow.add_node("insight", insight_agent)
    workflow.add_node("validator", validator_agent)
    workflow.add_node("narrator", narrator_agent)
    
    # Define the flow
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "explorer")
    workflow.add_edge("explorer", "sql")
    workflow.add_edge("sql", "insight")
    workflow.add_edge("insight", "validator")
    workflow.add_edge("validator", "narrator")
    workflow.add_edge("narrator", END)
    
    return workflow.compile()

def analyze_dataset(csv_path: str):
    """Main function to analyze a dataset (supports CSV and XLSX)"""
    
    print("=" * 60)
    print("🤖 Multi-Agent AI Data Analyst Platform")
    print("   📊 Enhanced with Advanced Data Processing")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found: {csv_path}")
        return
    
    # Load initial data summary using advanced loader
    try:
        print("\n🔄 Loading and processing data...")
        df, quality_report = load_and_process_data(csv_path)
        
        data_summary = f"""Dataset: {csv_path}
Rows: {len(df)}, Columns: {len(df.columns)}
Columns: {', '.join(df.columns.tolist())}
File Type: {csv_path.split('.')[-1].upper()}
Encoding: {quality_report['file_info'].get('encoding', 'N/A')}
Quality: {quality_report['data_quality'].get('duplicates_removed', 0)} duplicates removed"""
        
        print(f"✅ Data loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return
    
    # Initialize state
    initial_state = {
        "dataset_path": csv_path,
        "data_summary": data_summary,
        "quality_report": quality_report,
        "analysis_plan": "",
        "exploration_results": "",
        "sql_queries": "",
        "insights": "",
        "validation_results": "",
        "final_report": "",
        "messages": []
    }
    
    # Create and run the workflow
    app = create_workflow()
    
    print(f"\n📊 Analyzing: {csv_path}")
    print(f"📈 Dataset: {len(df)} rows × {len(df.columns)} columns\n")
    
    # Run the multi-agent analysis
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("\n" + "─" * 60)
    print("📋 EXECUTION LOG:")
    print("─" * 60)
    for msg in final_state['messages']:
        print(msg)
    
    print("\n" + "─" * 60)
    print("📝 FINAL REPORT:")
    print("─" * 60)
    print(final_state['final_report'])
    
    print("\n" + "─" * 60)
    print("💡 KEY INSIGHTS:")
    print("─" * 60)
    print(final_state['insights'])
    
    print("\n" + "=" * 60)
    print("✅ Analysis session complete!")
    print("=" * 60)
    
    return final_state


async def analyze_dataset_streaming(file_path: str) -> AsyncGenerator[dict, None]:
    """
    Analyze dataset with streaming progress updates for API
    Supports both CSV and XLSX files with advanced data processing
    Yields progress updates as dictionaries
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        yield {"type": "error", "message": f"File not found: {file_path}"}
        return
    
    # Load initial data summary with advanced processing
    try:
        yield {
            "type": "progress",
            "agent": "loader",
            "message": "Loading and processing your data...",
            "step": 0,
            "total_steps": 7
        }
        
        # Use advanced data loader
        df, quality_report = load_and_process_data(file_path)
        
        # Prepare comprehensive data summary
        data_summary = f"""Dataset: {file_path}
Rows: {len(df)}, Columns: {len(df.columns)}
Columns: {', '.join(df.columns.tolist())}
File Type: {file_path.split('.')[-1].upper()}
Encoding: {quality_report['file_info'].get('encoding', 'N/A')}
Quality: {quality_report['data_quality'].get('duplicates_removed', 0)} duplicates removed"""
        
        yield {
            "type": "start",
            "message": "Data loaded and processed successfully",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "file_type": file_path.split('.')[-1].upper(),
            "quality_report": {
                "duplicates_removed": quality_report['data_quality'].get('duplicates_removed', 0),
                "missing_values_handled": len(quality_report['data_quality'].get('missing_values', {})),
                "outliers_detected": len(quality_report['data_quality'].get('outliers', {}))
            }
        }
        
    except Exception as e:
        yield {"type": "error", "message": f"Error processing file: {str(e)}"}
        return
    
    # Initialize state
    initial_state = {
        "dataset_path": file_path,
        "data_summary": data_summary,
        "quality_report": quality_report,
        "analysis_plan": "",
        "exploration_results": "",
        "sql_queries": "",
        "insights": "",
        "validation_results": "",
        "final_report": "",
        "messages": []
    }
    
    # Create workflow
    app = create_workflow()
    
    # Agent progress tracking
    agents = [
        ("planner", "Analyzing your data structure..."),
        ("explorer", "Examining patterns and quality..."),
        ("sql", "Processing dataset operations..."),
        ("insight", "Extracting deep insights..."),
        ("validator", "Validating findings..."),
        ("narrator", "Preparing your comprehensive report...")
    ]
    
    current_agent_idx = 0
    
    # Custom agent execution with progress updates
    state = initial_state
    
    for idx, (agent_name, agent_message) in enumerate(agents):
        yield {
            "type": "progress",
            "agent": agent_name,
            "message": agent_message,
            "step": idx + 1,
            "total_steps": len(agents)
        }
        
        # Simulate agent work and get result
        await asyncio.sleep(0.5)  # Small delay for UI update
        
        # Execute the specific agent
        if agent_name == "planner":
            state = planner_agent(state)
        elif agent_name == "explorer":
            state = explorer_agent(state)
        elif agent_name == "sql":
            state = sql_agent(state)
        elif agent_name == "insight":
            state = insight_agent(state)
        elif agent_name == "validator":
            state = validator_agent(state)
        elif agent_name == "narrator":
            state = narrator_agent(state)
    
    # Send final results with quality info
    user_summary = extract_user_friendly_summary(state, file_path)
    user_summary['data_quality'] = {
        "duplicates_removed": quality_report['data_quality'].get('duplicates_removed', 0),
        "missing_values_columns": list(quality_report['data_quality'].get('missing_values', {}).keys()),
        "outliers_columns": list(quality_report['data_quality'].get('outliers', {}).keys()),
        "file_encoding": quality_report['file_info'].get('encoding', 'UTF-8')
    }
    
    # Generate dashboard data for visualizations
    try:
        dashboard_data = generate_dashboard_data(df, quality_report)
        user_summary['dashboard'] = dashboard_data
    except Exception as e:
        print(f"Dashboard generation warning: {str(e)}")
        user_summary['dashboard'] = {"charts": [], "metrics": []}
    
    yield {
        "type": "complete",
        "message": "Advanced analysis complete",
        "results": user_summary
    }


# Example usage
if __name__ == "__main__":
    # Example: Analyze a CSV file
    # Replace with your actual CSV file path
    
    csv_file = "sample_data.csv"
    
    # Create a sample dataset if it doesn't exist
    if not os.path.exists(csv_file):
        print("📝 Creating sample dataset...")
        sample_data = pd.DataFrame({
            'product': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
            'sales': [100, 150, 200, 120, 160, 210] * 10,
            'revenue': [1000, 2250, 4000, 1200, 2560, 4410] * 10,
            'region': ['North', 'South', 'East', 'West', 'North', 'South'] * 10
        })
        sample_data.to_csv(csv_file, index=False)
        print(f"✅ Sample dataset created: {csv_file}")
    
    # Run the analysis
    analyze_dataset(csv_file)
