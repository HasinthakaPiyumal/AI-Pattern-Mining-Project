from shopping_assistant import ShoppingAssistant
import os

# Ensure the plugins directory exists and dummy plugins are created for demonstration
if not os.path.exists("plugins"):
    os.makedirs("plugins")

# Create dummy plugin files if they don't exist
# This assumes the content of these files was previously generated and is correct
if not os.path.exists("plugins/price_comparison_plugin.py"):
    with open("plugins/price_comparison_plugin.py", "w") as f:
        f.write("""
class PriceComparisonPlugin:
    def execute(self, product_name, max_price=None):
        print(f"[PLUGIN] PriceComparisonPlugin: Comparing prices for '{product_name}'...")
        mock_prices = {
            "laptop x": {"Site A": 999.00, "Site B": 985.50, "Site C": 1010.00},
            "gaming mouse": {"Site A": 75.00, "Site B": 72.99, "Site C": 78.00},
            "smartphone y": {"Site A": 600.00, "Site B": 595.00, "Site C": 605.00},
        }
        product_name_lower = product_name.lower()
        if product_name_lower in mock_prices:
            prices = mock_prices[product_name_lower]
            best_price = min(prices.values())
            best_site = [site for site, price in prices.items() if price == best_price][0]
            if max_price is not None and best_price > max_price:
                return f"For '{product_name}': Best price found is ${best_price:.2f} at {best_site}, which is above your budget of ${max_price:.2f}."
            else:
                return f"For '{product_name}': Best price found is ${best_price:.2f} at {best_site}."
        else:
            return f"Price comparison not available for '{product_name}'."
""")

if not os.path.exists("plugins/product_reviewer_plugin.py"):
    with open("plugins/product_reviewer_plugin.py", "w") as f:
        f.write("""
class ProductReviewerPlugin:
    def execute(self, product_name):
        print(f"[PLUGIN] ProductReviewerPlugin: Summarizing reviews for '{product_name}'...")
        mock_reviews = {
            "laptop x": "Overall positive, users praise its performance and battery life, though some mention screen glare.",
            "gaming mouse": "Excellent responsiveness and comfortable grip, but software customization can be complex.",
            "smartphone y": "Camera quality is a standout feature, and the display is vibrant. Battery life is average.",
        }
        product_name_lower = product_name.lower()
        if product_name_lower in mock_reviews:
            return f"Review Summary for '{product_name}': {mock_reviews[product_name_lower]}"
        else:
            return f"Review summary not available for '{product_name}'."
""")


assistant = ShoppingAssistant()

# Example requests
requests = [
    "What is the price of Laptop X under $900?",
    "Summarize reviews for Gaming Mouse.",
    "Find laptops under $950",
    "Compare price for Smartphone Y",
    "Tell me about smartwatches.", # This should trigger the 'none' action
    "What is the price of Laptop X?"
]

for req in requests:
    response = assistant.process_request(req)
    print(f"Assistant Response: {response}\n" + "-"*50)

print("\nDemonstration complete. Check the output above for plugin and code generation actions.")
