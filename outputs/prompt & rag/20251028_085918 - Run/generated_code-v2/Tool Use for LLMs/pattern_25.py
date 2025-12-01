import inspect

class ECommercePlugin:
    def execute(self, action: str, **kwargs):
        raise NotImplementedError("Plugins must implement an execute method.")

class AmazonPlugin(ECommercePlugin):
    def search_product(self, product_name: str):
        return f"Searching Amazon for '{product_name}'... (simulated result)"

    def compare_price(self, product_name: str):
        return f"Comparing prices for '{product_name}' on Amazon... (simulated result)"

    def add_to_wishlist(self, product_name: str):
        return f"Adding '{product_name}' to Amazon wishlist... (simulated result)"

    def execute(self, action: str, **kwargs):
        if hasattr(self, action):
            method = getattr(self, action)
            return method(**kwargs)
        return f"AmazonPlugin does not support action '{action}'."

class eBayPlugin(ECommercePlugin):
    def search_product(self, product_name: str):
        return f"Searching eBay for '{product_name}'... (simulated result)"

    def compare_price(self, product_name: str):
        return f"Comparing prices for '{product_name}' on eBay... (simulated result)"

    def execute(self, action: str, **kwargs):
        if hasattr(self, action):
            method = getattr(self, action)
            return method(**kwargs)
        return f"eBayPlugin does not support action '{action}'."

class PaymentPlugin(ECommercePlugin):
    def automate_purchase(self, product_name: str, quantity: int, platform: str):
        return f"Automating purchase of {quantity}x '{product_name}' on {platform} via PaymentPlugin... (simulated result)"

    def execute(self, action: str, **kwargs):
        if hasattr(self, action):
            method = getattr(self, action)
            return method(**kwargs)
        return f"PaymentPlugin does not support action '{action}'."


class PluginManager:
    def __init__(self):
        self._plugins = {}

    def register_plugin(self, name: str, plugin_instance: ECommercePlugin):
        self._plugins[name.lower()] = plugin_instance

    def get_plugin(self, name: str):
        return self._plugins.get(name.lower())


class ECommerceAssistant:
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager

    def process_request(self, request: str):
        request = request.lower()

        # Simplified intent recognition and entity extraction
        if "search" in request:
            plugin_name = None
            if "amazon" in request: plugin_name = "amazon"
            elif "ebay" in request: plugin_name = "ebay"
            
            product_name = self._extract_product_name(request)
            
            if plugin_name and product_name:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if plugin: return plugin.execute("search_product", product_name=product_name)
                else: return f"Error: Plugin '{plugin_name}' not found."
            else: return "Please specify a product to search and optionally a platform (Amazon/eBay)."

        elif "compare prices" in request:
            plugin_name = None
            if "amazon" in request: plugin_name = "amazon"
            elif "ebay" in request: plugin_name = "ebay"

            product_name = self._extract_product_name(request)

            if plugin_name and product_name:
                plugin = self.plugin_manager.get_plugin(plugin_name)
                if plugin: return plugin.execute("compare_price", product_name=product_name)
                else: return f"Error: Plugin '{plugin_name}' not found."
            else: return "Please specify a product to compare prices and a platform (Amazon/eBay)."

        elif "add to wishlist" in request and "amazon" in request:
            product_name = self._extract_product_name(request)
            if product_name:
                plugin = self.plugin_manager.get_plugin("amazon")
                if plugin: return plugin.execute("add_to_wishlist", product_name=product_name)
                else: return "Error: Amazon plugin not found."
            else: return "Please specify a product to add to the Amazon wishlist."

        elif "buy" in request or "purchase" in request:
            product_name = self._extract_product_name(request)
            quantity = self._extract_quantity(request)
            platform = None
            if "amazon" in request: platform = "Amazon"
            elif "ebay" in request: platform = "eBay"

            if product_name and quantity and platform:
                plugin = self.plugin_manager.get_plugin("payment")
                if plugin: return plugin.execute("automate_purchase", product_name=product_name, quantity=quantity, platform=platform)
                else: return "Error: Payment plugin not found."
            else: return "Please specify a product, quantity, and platform to purchase."

        return "I'm sorry, I don't understand that request."

    def _extract_product_name(self, request: str):
        keywords = ["for", "of", "a", "an", "the"]
        parts = request.split()
        product_words = []
        start_collecting = False

        for i, part in enumerate(parts):
            if part in ["search", "compare", "add", "buy", "purchase"]:
                if i + 1 < len(parts) and parts[i + 1] in keywords:
                    start_collecting = True
                    continue
            if start_collecting and part not in ["on", "amazon", "ebay", "wishlist"] and not part.isdigit():
                product_words.append(part)
            elif start_collecting and part in ["on", "amazon", "ebay", "wishlist"]:
                break
            
        return " ".join(product_words).strip()

    def _extract_quantity(self, request: str):
        import re
        match = re.search(r'(\\d+)\s*(?:units?|items?|pieces?)', request)
        if match: return int(match.group(1))
        match = re.search(r'buy\s*(\\d+)', request)
        if match: return int(match.group(1))
        return 1 # Default to 1 if quantity not specified


if __name__ == "__main__":
    # Setup Plugin Manager and register plugins
    plugin_manager = PluginManager()
    plugin_manager.register_plugin("amazon", AmazonPlugin())
    plugin_manager.register_plugin("ebay", eBayPlugin())
    plugin_manager.register_plugin("payment", PaymentPlugin())

    # Initialize E-commerce Assistant
    assistant = ECommerceAssistant(plugin_manager)

    # Test cases
    print(assistant.process_request("Search for a smart watch on Amazon"))
    print(assistant.process_request("Compare prices for 'wireless headphones' on eBay"))
    print(assistant.process_request("Add 'portable charger' to my Amazon wishlist"))
    print(assistant.process_request("Buy 2 smart home devices on Amazon"))
    print(assistant.process_request("Purchase a new laptop on eBay"))
    print(assistant.process_request("What is the weather like?"))
    print(assistant.process_request("Search for drone")) # Missing platform
    print(assistant.process_request("Add to my Amazon wishlist")) # Missing product
    print(assistant.process_request("buy 5 books")) # Missing platform