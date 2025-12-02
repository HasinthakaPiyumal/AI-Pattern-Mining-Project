import pandas as pd
import io
import re
from PyPDF2 import PdfReader # Assuming PyPDF2 for PDF extraction

class FinancialAnalyzer:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_file):
        """
        Extracts text from a PDF file.
        """
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error extracting text from PDF: {e}"

    def simulate_llm_code_generation(self, financial_text: str, user_query: str) -> str:
        """
        Simulates an LLM generating Python code based on financial text and a user query.
        In a real scenario, this would involve an actual LLM call (e.g., OpenAI, Gemini).
        For demonstration, it generates sample code for common financial ratios.
        """
        
        # Placeholder for extracting relevant numbers from the financial_text
        # This part would be more sophisticated with a real LLM, using NER or RAG
        
        # Helper to find a value near a keyword
        def find_value(text, keywords):
            for keyword in keywords:
                # Look for patterns like 
                match = re.search(f"{re.escape(keyword)}\s*[-—–]*\s*([$]?\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{2}})?)", text, re.IGNORECASE)
                if match:
                    value = match.group(1).replace('$', '').replace(',', '')
                    return float(value)
            return None

        # Example: Simulating extraction for Debt-to-Equity
        if "debt-to-equity" in user_query.lower() or "debt to equity" in user_query.lower():
            total_debt = find_value(financial_text, ["Total Debt", "Long-term Debt", "Current Debt"])
            shareholder_equity = find_value(financial_text, ["Total Shareholder Equity", "Shareholders' Equity", "Total Equity"])
            
            if total_debt is not None and shareholder_equity is not None and shareholder_equity != 0:
                return f"total_debt = {total_debt}\nshareholder_equity = {shareholder_equity}\ndebt_to_equity = total_debt / shareholder_equity\nprint(f'Debt-to-Equity Ratio: {{debt_to_equity:.2f}}')"
            else:
                return "print('Could not find sufficient data to calculate Debt-to-Equity ratio.')"

        # Example: Simulating extraction for Profit Margin
        elif "profit margin" in user_query.lower():
            net_income = find_value(financial_text, ["Net Income", "Net Profit"])
            revenue = find_value(financial_text, ["Revenue", "Total Revenue", "Sales"])

            if net_income is not None and revenue is not None and revenue != 0:
                return f"net_income = {net_income}\nrevenue = {revenue}\nprofit_margin = (net_income / revenue) * 100\nprint(f'Profit Margin: {{profit_margin:.2f}}%')"
            else:
                return "print('Could not find sufficient data to calculate Profit Margin.')"

        else:
            return f"print('No specific code generation logic for the query: {user_query}.')"

    def execute_python_code(self, code: str) -> str:
        """
        Executes the given Python code in a sandboxed environment (simulated here).
        In a real application, consider secure execution environments (e.g., Docker, separate process).
        """
        output_capture = io.StringIO()
        try:
            # Redirect stdout to capture print statements
            import sys
            sys.stdout = output_capture
            exec(code)
            sys.stdout = sys.__stdout__ # Reset stdout
            return output_capture.getvalue()
        except Exception as e:
            sys.stdout = sys.__stdout__ # Ensure stdout is reset even on error
            return f"Error during code execution: {e}"

    def simulate_llm_report_generation(self, user_query: str, financial_text: str, code_output: str) -> str:
        """
        Simulates an LLM generating a natural language financial report based on the
        user query, original financial text, and the numerical results from code execution.
        """
        report = f"Based on your query: '{user_query}', and the provided financial document:\n\n"
        report += f"The system performed a computational analysis. Here are the key findings:\n\n"
        report += f"*   **Computational Result:** {code_output.strip()}\n\n"

        if "debt-to-equity" in user_query.lower():
            if "Debt-to-Equity Ratio" in code_output:
                ratio_value = re.search(r'Debt-to-Equity Ratio: (\d+\.\d+)', code_output)
                if ratio_value: 
                    ratio = float(ratio_value.group(1))
                    report += f"The calculated Debt-to-Equity ratio is **{ratio:.2f}**. "
                    if ratio < 1.0: report += "This generally indicates a lower risk company, as equity finances a larger portion of assets."
                    elif 1.0 <= ratio <= 2.0: report += "This ratio suggests a balanced approach to financing, with a moderate reliance on debt."
                    else: report += "A higher ratio indicates that the company relies heavily on debt financing, which can imply higher financial risk."
                else:
                    report += "The Debt-to-Equity ratio could not be precisely determined from the code output.\n"
            else:
                report += "The Debt-to-Equity ratio calculation was not successful or data was insufficient.\n"

        elif "profit margin" in user_query.lower():
            if "Profit Margin" in code_output:
                margin_value = re.search(r'Profit Margin: (\d+\.\d+)%', code_output)
                if margin_value:
                    margin = float(margin_value.group(1))
                    report += f"The calculated Profit Margin is **{margin:.2f}%**. "
                    if margin > 15: report += "This indicates strong profitability, with a significant portion of revenue converting to net income."
                    elif 5 <= margin <= 15: report += "This suggests a healthy profit margin, indicating efficient operations."
                    else: report += "A lower profit margin might indicate tight competition or higher operating costs."
                else:
                    report += "The Profit Margin could not be precisely determined from the code output.\n"
            else:
                report += "The Profit Margin calculation was not successful or data was insufficient.\n"
        else:
            report += "Further detailed natural language analysis based on the specific query and results would be provided by a sophisticated LLM.\n"

        report += "\n--- End of Report ---"
        return report
