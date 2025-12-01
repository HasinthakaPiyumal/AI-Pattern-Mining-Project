import re

class DataStore:
    def __init__(self):
        self.products = {}
        self.orders = {}
        self.customers = {}
        self.logs = []

    def add_product(self, platform, product_details):
        if platform not in self.products:
            self.products[platform] = {}
        product_id = f"{platform}_{len(self.products[platform]) + 1}"
        self.products[platform][product_id] = product_details
        self.logs.append(f"Added product {product_id} to {platform}")
        return product_id

    def update_product_inventory(self, platform, product_id, quantity):
        if platform in self.products and product_id in self.products[platform]:
            self.products[platform][product_id]["inventory"] = quantity
            self.logs.append(f"Updated inventory for {product_id} on {platform} to {quantity}")
            return True
        return False

    def add_order(self, platform, order_details):
        if platform not in self.orders:
            self.orders[platform] = {}
        order_id = f"{platform}_ORD_{len(self.orders[platform]) + 1}"
        self.orders[platform][order_id] = order_details
        self.logs.append(f"Added order {order_id} to {platform}")
        return order_id

    def update_order_status(self, platform, order_id, status):
        if platform in self.orders and order_id in self.orders[platform]:
            self.orders[platform][order_id]["status"] = status
            self.logs.append(f"Updated status for order {order_id} on {platform} to {status}")
            return True
        return False

    def add_customer_query(self, platform, customer_id, query):
        if platform not in self.customers:
            self.customers[platform] = {}
        if customer_id not in self.customers[platform]:
            self.customers[platform][customer_id] = []
        self.customers[platform][customer_id].append({"query": query, "response": None})
        self.logs.append(f"Received query from customer {customer_id} on {platform}")
        return True

    def respond_to_customer_query(self, platform, customer_id, query_index, response):
        if platform in self.customers and customer_id in self.customers[platform]:
            if 0 <= query_index < len(self.customers[platform][customer_id]):
                self.customers[platform][customer_id][query_index]["response"] = response
                self.logs.append(f"Responded to customer {customer_id} on {platform} for query index {query_index}")
                return True
        return False


class SemanticInterface:
    def parse_command(self, command):
        command = command.lower()
        intent = None
        platform = None
        params = {}

        # Extract platform
        platform_match = re.search(r"on (shopify|etsy|amazon|ebay)", command)
        if platform_match:
            platform = platform_match.group(1).capitalize()

        if "list product" in command or "add product" in command:
            intent = "list_product"
            product_name_match = re.search(r"product '([^']+)'", command)
            if product_name_match:
                params["product_name"] = product_name_match.group(1)
            price_match = re.search(r"price (\$\d+\.?\d*)", command)
            if price_match:
                params["price"] = price_match.group(1)
            inventory_match = re.search(r"inventory (\d+)", command)
            if inventory_match:
                params["inventory"] = int(inventory_match.group(1))

        elif "update inventory" in command:
            intent = "update_inventory"
            product_id_match = re.search(r"inventory for product (\w+_\d+)", command)
            if product_id_match:
                params["product_id"] = product_id_match.group(1)
            quantity_match = re.search(r"to (\d+)", command)
            if quantity_match:
                params["quantity"] = int(quantity_match.group(1))

        elif "process order" in command or "update order" in command:
            intent = "process_order"
            order_id_match = re.search(r"order (\w+_ORD_\d+)", command)
            if order_id_match:
                params["order_id"] = order_id_match.group(1)
            status_match = re.search(r"status to (\w+)", command)
            if status_match:
                params["status"] = status_match.group(1).capitalize()

        elif "respond to customer query" in command or "reply to customer" in command:
            intent = "respond_to_customer_query"
            customer_id_match = re.search(r"customer (\w+_\d+)", command)
            if customer_id_match:
                params["customer_id"] = customer_id_match.group(1)
            response_match = re.search(r"with \'([^']+)\'", command)
            if response_match:
                params["response"] = response_match.group(1)
            query_index_match = re.search(r"query index (\d+)", command)
            if query_index_match:
                params["query_index"] = int(query_index_match.group(1))

        elif "run script" in command and "for" in command:
            intent = "run_script"
            script_match = re.search(r"run script \'([^']+)\'", command)
            if script_match:
                params["script_code"] = script_match.group(1)


        return {"intent": intent, "platform": platform, "params": params}


class GUIInterface:
    def click(self, element_id):
        print(f"[GUI Interface] Simulating click on element: {element_id}")

    def type_text(self, element_id, text):
        print(f"[GUI Interface] Simulating typing '{text}' into element: {element_id}")

    def navigate_to(self, url):
        print(f"[GUI Interface] Navigating to URL: {url}")


class ProgrammingInterface:
    def execute_code(self, python_code):
        print(f"[Programming Interface] Executing custom Python code:\n---\n{python_code}\n---")
        try:
            # A restricted execution environment for safety (though not a full sandbox)
            exec(python_code, {"print": print, "__builtins__": {}})
            return "Code executed successfully."
        except Exception as e:
            return f"Error executing code: {e}"


class ShopifyAdapter:
    def __init__(self, datastore):
        self.datastore = datastore

    def list_product(self, product_details):
        print(f"[Shopify Adapter] Listing product: {product_details['product_name']} on Shopify")
        product_id = self.datastore.add_product("Shopify", product_details)
        return f"Product '{product_details['product_name']}' listed on Shopify with ID {product_id}."

    def update_inventory(self, product_id, quantity):
        print(f"[Shopify Adapter] Updating inventory for {product_id} on Shopify to {quantity}")
        if self.datastore.update_product_inventory("Shopify", product_id, quantity):
            return f"Inventory for product {product_id} updated to {quantity} on Shopify."
        return f"Product {product_id} not found on Shopify."

    def process_order(self, order_id, status):
        print(f"[Shopify Adapter] Processing order {order_id} on Shopify with status {status}")
        if self.datastore.update_order_status("Shopify", order_id, status):
            return f"Order {order_id} status updated to {status} on Shopify."
        return f"Order {order_id} not found on Shopify."

    def respond_to_customer_query(self, customer_id, query_index, response):
        print(f"[Shopify Adapter] Responding to customer {customer_id} query on Shopify with '{response}'")
        if self.datastore.respond_to_customer_query("Shopify", customer_id, query_index, response):
            return f"Responded to customer {customer_id} query on Shopify."
        return f"Customer {customer_id} or query index {query_index} not found on Shopify."


class EtsyAdapter:
    def __init__(self, datastore):
        self.datastore = datastore

    def list_product(self, product_details):
        print(f"[Etsy Adapter] Listing product: {product_details['product_name']} on Etsy")
        product_id = self.datastore.add_product("Etsy", product_details)
        return f"Product '{product_details['product_name']}' listed on Etsy with ID {product_id}."

    def update_inventory(self, product_id, quantity):
        print(f"[Etsy Adapter] Updating inventory for {product_id} on Etsy to {quantity}")
        if self.datastore.update_product_inventory("Etsy", product_id, quantity):
            return f"Inventory for product {product_id} updated to {quantity} on Etsy."
        return f"Product {product_id} not found on Etsy."

    def process_order(self, order_id, status):
        print(f"[Etsy Adapter] Processing order {order_id} on Etsy with status {status}")
        if self.datastore.update_order_status("Etsy", order_id, status):
            return f"Order {order_id} status updated to {status} on Etsy."
        return f"Order {order_id} not found on Etsy."

    def respond_to_customer_query(self, customer_id, query_index, response):
        print(f"[Etsy Adapter] Responding to customer {customer_id} query on Etsy with '{response}'")
        if self.datastore.respond_to_customer_query("Etsy", customer_id, query_index, response):
            return f"Responded to customer {customer_id} query on Etsy."
        return f"Customer {customer_id} or query index {query_index} not found on Etsy."


class AmazonAdapter:
    def __init__(self, datastore):
        self.datastore = datastore

    def list_product(self, product_details):
        print(f"[Amazon Adapter] Listing product: {product_details['product_name']} on Amazon")
        product_id = self.datastore.add_product("Amazon", product_details)
        return f"Product '{product_details['product_name']}' listed on Amazon with ID {product_id}."

    def update_inventory(self, product_id, quantity):
        print(f"[Amazon Adapter] Updating inventory for {product_id} on Amazon to {quantity}")
        if self.datastore.update_product_inventory("Amazon", product_id, quantity):
            return f"Inventory for product {product_id} updated to {quantity} on Amazon."
        return f"Product {product_id} not found on Amazon."

    def process_order(self, order_id, status):
        print(f"[Amazon Adapter] Processing order {order_id} on Amazon with status {status}")
        if self.datastore.update_order_status("Amazon", order_id, status):
            return f"Order {order_id} status updated to {status} on Amazon."
        return f"Order {order_id} not found on Amazon."

    def respond_to_customer_query(self, customer_id, query_index, response):
        print(f"[Amazon Adapter] Responding to customer {customer_id} query on Amazon with '{response}'")
        if self.datastore.respond_to_customer_query("Amazon", customer_id, query_index, response):
            return f"Responded to customer {customer_id} query on Amazon."
        return f"Customer {customer_id} or query index {query_index} not found on Amazon."


class EbayAdapter:
    def __init__(self, datastore):
        self.datastore = datastore

    def list_product(self, product_details):
        print(f"[eBay Adapter] Listing product: {product_details['product_name']} on eBay")
        product_id = self.datastore.add_product("eBay", product_details)
        return f"Product '{product_details['product_name']}' listed on eBay with ID {product_id}."

    def update_inventory(self, product_id, quantity):
        print(f"[eBay Adapter] Updating inventory for {product_id} on eBay to {quantity}")
        if self.datastore.update_product_inventory("eBay", product_id, quantity):
            return f"Inventory for product {product_id} updated to {quantity} on eBay."
        return f"Product {product_id} not found on eBay."

    def process_order(self, order_id, status):
        print(f"[eBay Adapter] Processing order {order_id} on eBay with status {status}")
        if self.datastore.update_order_status("eBay", order_id, status):
            return f"Order {order_id} status updated to {status} on eBay."
        return f"Order {order_id} not found on eBay."

    def respond_to_customer_query(self, customer_id, query_index, response):
        print(f"[eBay Adapter] Responding to customer {customer_id} query on eBay with '{response}'")
        if self.datastore.respond_to_customer_query("eBay", customer_id, query_index, response):
            return f"Responded to customer {customer_id} query on eBay."
        return f"Customer {customer_id} or query index {query_index} not found on eBay."


class UniversalECommerceAssistant:
    def __init__(self):
        self.datastore = DataStore()
        self.semantic_interface = SemanticInterface()
        self.gui_interface = GUIInterface()
        self.programming_interface = ProgrammingInterface()
        self.platform_adapters = {
            "Shopify": ShopifyAdapter(self.datastore),
            "Etsy": EtsyAdapter(self.datastore),
            "Amazon": AmazonAdapter(self.datastore),
            "eBay": EbayAdapter(self.datastore),
        }

    def process_command(self, command):
        parsed_command = self.semantic_interface.parse_command(command)
        intent = parsed_command["intent"]
        platform = parsed_command["platform"]
        params = parsed_command["params"]

        if intent == "run_script":
            script_code = params.get("script_code")
            if script_code:
                return self.programming_interface.execute_code(script_code)
            else:
                return "Error: No script code provided for 'run script' command."

        if not platform or platform not in self.platform_adapters:
            return f"Error: Unsupported platform or platform not specified in command: {command}"

        adapter = self.platform_adapters[platform]

        if intent == "list_product":
            return adapter.list_product(params)
        elif intent == "update_inventory":
            return adapter.update_inventory(params["product_id"], params["quantity"])
        elif intent == "process_order":
            return adapter.process_order(params["order_id"], params["status"])
        elif intent == "respond_to_customer_query":
            # For simplicity, we assume the assistant knows the customer_id and query_index for a response
            # In a real system, this would involve retrieving active queries for a customer
            if "customer_id" in params and "response" in params:
                # Simulating a query being added first for response
                self.datastore.add_customer_query(platform, params["customer_id"], f"Simulated query for {params['customer_id']}")
                # For demo, assuming the latest query is to be responded to (index -1 if add_customer_query is consistent)
                query_index = params.get("query_index", len(self.datastore.customers[platform][params['customer_id']]) - 1)
                return adapter.respond_to_customer_query(params["customer_id"], query_index, params["response"])
            else:
                return "Error: Missing customer ID or response for customer query."
        else:
            return f"Error: Unknown intent '{intent}' or invalid command: {command}"


if __name__ == "__main__":
    assistant = UniversalECommerceAssistant()

    print("--- Demonstrating Universal E-commerce Assistant ---")

    # Semantic Interface & Platform Adapter interactions
    print("\n--- Listing Products ---")
    print(assistant.process_command("List my new product 'Handcrafted Mug' with price $18.99 and inventory 10 on Shopify."))
    print(assistant.process_command("Add product 'Vintage Necklace' with price $45.00 and inventory 5 on Etsy."))
    print(assistant.process_command("List product 'Smart Home Hub' with price $129.99 on Amazon."))
    print(assistant.process_command("Add product 'Collectible Comic' with price $9.99 on eBay."))

    print("\n--- Updating Inventory ---")
    shopify_product_id = list(assistant.datastore.products['Shopify'].keys())[0]
    etsy_product_id = list(assistant.datastore.products['Etsy'].keys())[0]
    print(assistant.process_command(f"Update inventory for product {shopify_product_id} to 8 on Shopify."))
    print(assistant.process_command(f"Update inventory for product {etsy_product_id} to 3 on Etsy."))

    print("\n--- Processing Orders ---")
    # Simulate adding an order first
    assistant.datastore.add_order("Shopify", {"items": [{"id": "item1", "qty": 1}], "customer": "cust1", "status": "pending"})
    shopify_order_id = list(assistant.datastore.orders['Shopify'].keys())[0]
    print(assistant.process_command(f"Process order {shopify_order_id} with status to Shipped on Shopify."))

    assistant.datastore.add_order("Amazon", {"items": [{"id": "item2", "qty": 2}], "customer": "cust2", "status": "pending"})
    amazon_order_id = list(assistant.datastore.orders['Amazon'].keys())[0]
    print(assistant.process_command(f"Update order {amazon_order_id} status to Delivered on Amazon."))

    print("\n--- Responding to Customer Queries ---")
    # Simulate a customer query before responding
    assistant.datastore.add_customer_query("Shopify", "Shopify_1", "Where is my order?")
    print(assistant.process_command("Respond to customer Shopify_1 with 'Your order is on its way!' on Shopify."))

    assistant.datastore.add_customer_query("Etsy", "Etsy_1", "Can I change the color?")
    print(assistant.process_command("Respond to customer Etsy_1 with 'Please send us a message with your preferred color change.' on Etsy."))


    print("\n--- Programming Interface (Custom Script) ---")
    custom_script = """
def calculate_discount(price, percentage):
    return price * (1 - percentage / 100)

price = 100
discount = 10
final_price = calculate_discount(price, discount)
print(f"Original price: {price}, Discount: {discount}%, Final price: {final_price}")
"""
    print(assistant.process_command(f"Run script '{custom_script}' for custom price calculation."))

    print("\n--- Data Store State ---")
    print("\nProducts:", assistant.datastore.products)
    print("\nOrders:", assistant.datastore.orders)
    print("\nCustomers:", assistant.datastore.customers)
    print("\nLogs:", assistant.datastore.logs)
