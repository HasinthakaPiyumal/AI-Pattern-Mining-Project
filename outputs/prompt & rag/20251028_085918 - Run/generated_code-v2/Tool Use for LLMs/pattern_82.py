import time
import concurrent.futures

class CustomerSupportAgent:
    def __init__(self, max_workers=5):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def _get_order_status(self, order_id):
        time.sleep(1) 
        if order_id == "ORDER123":
            return {"order_id": order_id, "status": "Shipped", "delivery_date": "2023-11-15"}
        return {"order_id": order_id, "status": "Not Found"}

    def _get_product_details(self, product_name):
        time.sleep(0.8) 
        if product_name == "Laptop":
            return {"product_name": product_name, "price": "$1200", "specs": "Intel i7, 16GB RAM"}
        return {"product_name": product_name, "details": "Not Found"}

    def _get_customer_history(self, customer_email):
        time.sleep(1.2) 
        if customer_email == "test@example.com":
            return {"customer_email": customer_email, "loyalty_tier": "Gold", "recent_purchases": ["Laptop", "Mouse"]}
        return {"customer_email": customer_email, "history": "Not Found"}

    def _simulate_inquiry_processing(self, inquiry):
        tasks = []
        context = {}

        if "order status" in inquiry.lower():
            order_id = "ORDER123" if "ORDER123" in inquiry else "UNKNOWN_ORDER"
            tasks.append(("get_order_status", order_id))
            context["order_id"] = order_id

        if "product details" in inquiry.lower():
            product_name = "Laptop" if "laptop" in inquiry.lower() else "UNKNOWN_PRODUCT"
            tasks.append(("get_product_details", product_name))
            context["product_name"] = product_name

        if "customer history" in inquiry.lower() or "my past purchases" in inquiry.lower():
            customer_email = "test@example.com" if "test@example.com" in inquiry else "UNKNOWN_EMAIL"
            tasks.append(("get_customer_history", customer_email))
            context["customer_email"] = customer_email
            
        return tasks, context

    def process_inquiry(self, inquiry):
        print(f"\nProcessing inquiry: '{inquiry}'")
        start_time = time.time()

        subtasks, context = self._simulate_inquiry_processing(inquiry)
        print(f"Identified subtasks: {subtasks}")

        futures = {}
        for task_name, arg in subtasks:
            if task_name == "get_order_status":
                futures[self.executor.submit(self._get_order_status, arg)] = task_name
            elif task_name == "get_product_details":
                futures[self.executor.submit(self._get_product_details, arg)] = task_name
            elif task_name == "get_customer_history":
                futures[self.executor.submit(self._get_customer_history, arg)] = task_name

        results = {}
        for future in concurrent.futures.as_completed(futures):
            task_name = futures[future]
            try:
                data = future.result()
                results[task_name] = data
                print(f"Completed {task_name}: {data}")
            except Exception as exc:
                print(f"{task_name} generated an exception: {exc}")
        
        response = self._synthesize_response(inquiry, results)
        end_time = time.time()
        print(f"Inquiry processed in {end_time - start_time:.2f} seconds.\n")
        return response

    def _synthesize_response(self, original_inquiry, task_results):
        response_parts = [f"Regarding your inquiry about '{original_inquiry}':"]

        if "get_order_status" in task_results:
            order_info = task_results["get_order_status"]
            if order_info["status"] != "Not Found":
                response_parts.append(f"Order {order_info['order_id']} is {order_info['status']} and expected by {order_info['delivery_date']}.")
            else:
                response_parts.append(f"Could not find details for order {order_info['order_id']}.")

        if "get_product_details" in task_results:
            product_info = task_results["get_product_details"]
            if product_info["details"] != "Not Found":
                response_parts.append(f"The {product_info['product_name']} costs {product_info['price']} with specs: {product_info['specs']}.")
            else:
                response_parts.append(f"Could not find details for product {product_info['product_name']}.")

        if "get_customer_history" in task_results:
            history_info = task_results["get_customer_history"]
            if history_info["history"] != "Not Found":
                response_parts.append(f"As a {history_info['loyalty_tier']} customer, your recent purchases include: {', '.join(history_info['recent_purchases'])}.")
            else:
                response_parts.append(f"Could not retrieve customer history for {history_info['customer_email']}.")
        
        if not task_results:
            response_parts.append("I'm sorry, I couldn't find relevant information for your request.")

        return "\n".join(response_parts)


if __name__ == "__main__":
    agent = CustomerSupportAgent()

    inquiry1 = "What is the status of ORDER123 and can you tell me about the Laptop?"
    response1 = agent.process_inquiry(inquiry1)
    print(response1)

    inquiry2 = "I want to know about the product Laptop and my past purchases (test@example.com)."
    response2 = agent.process_inquiry(inquiry2)
    print(response2)

    inquiry3 = "What is the status of UNKNOWN_ORDER?"
    response3 = agent.process_inquiry(inquiry3)
    print(response3)

    inquiry4 = "Just a simple greeting."
    response4 = agent.process_inquiry(inquiry4)
    print(response4)
